# Musepack (`.mpc`) — SV8 and SV7 field reference

Reference material for `tags/src/mpc.rs`.

## Vendored, verbatim

- **`libmpc-streaminfo.c`** — `streaminfo_read_header_sv7`, `streaminfo_read_header_sv8`,
  `check_streaminfo`, and the `samplefreqs` table. This is the normative reader for both
  stream versions.
- **`libmpc-streaminfo.h`** — the `mpc_streaminfo` structure the above fills.
- **`libmpc-mpcdec.h`** — `MPC_FRAME_LENGTH` and `MPC_DECODER_SYNTH_DELAY`, the two constants
  the SV7 sample-count arithmetic turns on.
- **`libmpc-COPYING.txt`** — the licence.

All retrieved **2026-07-31** from a clean mirror of the musepack.net sources,
`https://raw.githubusercontent.com/evpobr/libmpc/master/{libmpcdec/streaminfo.c,
include/mpc/streaminfo.h,include/mpc/mpcdec.h,libmpcdec/COPYING}`, unmodified.

**Licence status: redistributable.** BSD-3-Clause, "Copyright (c) 2005, The Musepack
Development Team. All rights reserved." — the full text is at the head of each vendored
source file *and* in `libmpc-COPYING.txt`, which satisfies the source-redistribution
condition.

## Consulted but *not* vendored — and why

The canonical prose specification is the Musepack project's Trac wiki:

- `http://trac.musepack.net/musepack/wiki/SV8Specification`
- `http://trac.musepack.net/musepack/wiki/SV7Specification`

**The host is dead** — `trac.musepack.net` refuses connections as of 2026-07-31. Both pages
were recovered from the Internet Archive on that date at
`https://web.archive.org/web/20251113105851id_/http://trac.musepack.net/musepack/wiki/SV8Specification`
and
`https://web.archive.org/web/20221207225144id_/http://trac.musepack.net/musepack/wiki/SV7Specification`.

They carry **no licence statement of any kind**, so they are cited rather than committed —
which is the `mp4/spec/NOTES.md` treatment for material whose redistribution is doubtful. The
BSD-licensed reference reader above states the same layout normatively and *is* committed, so
nothing below depends on a page that may not be re-fetchable. Short passages are quoted with
attribution.

Clean-room: written from the wiki pages and the vendored reference sources. No tag library was
consulted.

## Sniffing

- **SV8**: `MPCK` = `4D 50 43 4B` at the start of the stream.
- **SV7**: `MP+` = `4D 50 2B` followed by a version octet whose **low nibble is the stream
  version** and must be 7. So the fourth octet is typically `0x07` or `0x17`.

Both may sit behind a leading ID3v2 tag, which the reference reader skips before looking.

## Shared: the sample-frequency table

`libmpc-streaminfo.c` line 53 — the same four rates for both stream versions:

```c
static const mpc_int32_t samplefreqs[8] = { 44100, 48000, 37800, 32000 };
```

| index | rate  | index | rate    |
|-------|-------|-------|---------|
| 0     | 44100 | 2     | 37800   |
| 1     | 48000 | 3     | 32000   |

The array is declared with **eight** slots but only four initialisers, so indices 4–7 are
zero-filled; `check_streaminfo` then rejects `sample_freq == 0`. SV8's index field is three
bits wide and can therefore express 4–7 — **those values are undefined** and must be treated
as invalid, not clamped. SV7's field is only two bits, so it cannot go wrong.

`check_streaminfo` in full, the sanity gate both paths end with:

```c
if (si->max_band == 0 || si->max_band >= 32
    || si->channels > 2 || si->channels == 0 || si->sample_freq == 0)
        return MPC_STATUS_FAIL;
```

(libmpc supports at most two channels even though SV8's field allows sixteen.)

---

# SV8

## Packet framing

Everything after the `MPCK` magic is a flat sequence of packets. The wiki: *"All fields,
unless explicitly specified otherwise, are read and written in Big-Endian order."*

```
  [ key: 2 ASCII bytes ][ size: variable-length integer ][ payload ]
```

- **Key**: two characters, both of which must lie in `A`…`Z` (0x41–0x5A). That is the
  format's own validity/resynchronisation test (`mpc_check_key`), and it is what lets a
  parser tell a real packet from noise.
- **Size**: a big-endian base-128 varint. **Seven payload bits per octet in the low bits;
  the MSB (`0x80`) is the continuation flag** — set means another octet follows, clear means
  this is the last. Accumulate `v = (v << 7) | (b & 0x7F)`, most significant group first.
  At most 9 octets (63 bits).
- **The size value INCLUDES the two key octets and the size octets themselves.** The wiki:
  *"Size defines the packet length in bytes, including the Key and Size fields. So the
  minimum length of a block is 3 bytes."* A payload is therefore
  `size − 2 − (number of size octets)`.
- **Only the packet size is self-inclusive.** Every other varint in a payload — the `SH`
  sample count and beginning silence, the `SO` offset, the `ST` counts — is a plain value.
  The encoder distinguishes the two with an `addCodeSize` argument; a parser that applies the
  self-inclusive rule to the sample count would under-report the length of every file.
- A payload may be zero-length, and **may be zero-padded past its defined fields** ("All
  unused bits in a packet MUST be null"). Always advance by the size field, never by the sum
  of the field widths.

**Reserved keys**: `SH` stream header (mandatory), `RG` replaygain (mandatory), `EI` encoder
info, `SO` seek-table offset, `AP` audio packet (mandatory), `ST` seek table, `CT` chapter tag
(beta), `SE` stream end (mandatory, size exactly 3).

## `SH` — the stream header

Must precede the first `AP`. Bit-packed, MSB-first:

| # | field                    | width                | notes                                                     |
|---|--------------------------|----------------------|-----------------------------------------------------------|
| 1 | CRC32                    | 32 bits              | over the rest of the payload                              |
| 2 | stream version           | 8 bits               | must be 8                                                 |
| 3 | sample count             | **varint**, 1–9 oct. | plain value; 0 means unknown                              |
| 4 | beginning silence        | **varint**, 1–9 oct. | samples to discard at the start                            |
| 5 | sample frequency index   | **3 bits**           | the table above                                            |
| 6 | max used bands           | **5 bits**           | stored as value − 1; real value 1…32                      |
| 7 | channel count            | **4 bits**           | stored as value − 1; real value 1…16                      |
| 8 | mid/side used            | **1 bit**            |                                                            |
| 9 | audio block frames       | **3 bits**           | frames per `AP` = `4^value`; libmpc keeps `block_pwr = 2 × value` |

Fields 5–9 occupy exactly 16 bits, so a typical `SH` payload is 12 octets (4 CRC + 1 version +
4 sample count + 1 silence + 2 packed) and the whole packet 15.

The **`− 1` bias on fields 6 and 7 is the easiest thing to get wrong**: a two-channel file
stores `1`, so a parser that reads the field literally reports mono for every stereo file and
never trips a sanity check.

**CRC**: CRC-32/ISO-HDLC (the PNG CRC — reflected polynomial `0xEDB88320`, init all ones,
final XOR all ones), computed over the payload **from octet 4 to the end**, i.e.
`payload_len − 4` octets. Because the encoder may pad `SH` when it rewrites the packet at
finalisation, the padding is inside the CRC — so the length must come from the packet size,
not from the field widths. This scanner does not verify the CRC: a tag scan reports what a
file says about itself and lets a decoder be the one to refuse.

### Duration

```
  playable samples = sample_count − beginning_silence
  duration         = playable / sample_frequency
```

which is `mpc_streaminfo_get_length` verbatim. **Exact** — a declared sample count.

## Other packets (not read by this scanner, recorded for completeness)

`RG` — `version:8` (must be 1), then title gain, title peak, album gain, album peak as four
16-bit values, dB in Q8.8. `EI` — `profile:7` (quality in 4.3 fixed point), `PNS:1`, then
major/minor/build octets. `SO` — one varint: the byte offset from the start of the `SO`
packet to the start of the `ST` packet. `ST` — the seek table; entries after the first two are
second-difference coded and Golomb coded with `M = 2^12`. `CT` — a chapter tag: a sample
offset varint, gain and peak, then an APEv2 tag **with the `APETAGEX` preamble stripped**.
`SE` — three octets, the last packet of the stream; tags follow it.

---

# SV7

## Byte order — the part that is easy to get wrong

SV7 is **a stream of 32-bit little-endian words on disk that are decoded MSB-first after a
byte swap**. libmpc reads with `MPC_BUFFER_SWAP` and then runs an MSB-first bit reader over
the swapped words. In practice: read each aligned 4-octet group as a little-endian `u32` and
take fields from the top bit down.

## Header — 28 octets

The first word is raw octets, not swapped:

| off | field                                                                        |
|-----|------------------------------------------------------------------------------|
| 0–2 | magic `MP+` = `4D 50 2B`                                                      |
| 3   | **low nibble** = stream version, must be 7; **high nibble** = minor / PNS flag |

Then, with `W_k` = the little-endian `u32` at octet offset `4k`:

| word | field              | extraction            | notes                                            |
|------|--------------------|-----------------------|--------------------------------------------------|
| W1   | **FrameCount**     | all 32 bits           | count of 1152-sample frames                       |
| W2   | IntensityStereo    | `(W2 >> 31) & 1`      | should be 0                                       |
| W2   | MidSideStereo      | `(W2 >> 30) & 1`      |                                                   |
| W2   | MaxBand            | `(W2 >> 24) & 0x3F`   | 6 bits                                            |
| W2   | Profile            | `(W2 >> 20) & 0xF`    | 7 Telephone … 10 Standard, 11 Xtreme, 12 Insane   |
| W2   | Link               | `(W2 >> 18) & 3`      |                                                   |
| W2   | **SampleFreq idx** | `(W2 >> 16) & 3`      | 2 bits — the table above                          |
| W2   | MaxLevel           | `W2 & 0xFFFF`         | peak level of the input PCM                       |
| W3   | TitleGain / Peak   | `W3 >> 16` / `W3 & 0xFFFF` | gain is signed millibel                     |
| W4   | AlbumGain / Peak   | `W4 >> 16` / `W4 & 0xFFFF` |                                             |
| W5   | **TrueGapless**    | `(W5 >> 31) & 1`      |                                                   |
| W5   | **LastFrameLength**| `(W5 >> 20) & 0x7FF`  | 11 bits; 0 when TrueGapless is 0                  |
| W5   | FastSeek-safe      | `(W5 >> 19) & 1`      |                                                   |
| W6   | EncoderVersion     | `W6 >> 24`            | version × 100 (106 = 1.06)                        |

**Channel count is not stored.** libmpc hard-codes `si->channels = 2` for SV7 — every SV7
file is stereo.

Only 200 of the header's bits are header (32 magic + 5 × 32 + 8); the last 24 bits of W6 are
already audio. "28 octets" is the right amount to *read*, not the size of the header proper.

### Duration — and why it is exact more often than one would guess

`streaminfo_read_header_sv7`, verbatim:

```c
if (last_frame_samples == 0) last_frame_samples = MPC_FRAME_LENGTH;
else if (last_frame_samples > MPC_FRAME_LENGTH) return MPC_STATUS_FAIL;
si->samples = (mpc_int64_t) frames * MPC_FRAME_LENGTH;
if (si->is_true_gapless)
        si->samples -= (MPC_FRAME_LENGTH - last_frame_samples);
else
        si->samples -= MPC_DECODER_SYNTH_DELAY;
```

with `MPC_FRAME_LENGTH = 36 * 32 = 1152` and `MPC_DECODER_SYNTH_DELAY = 481`
(`libmpc-mpcdec.h`). So:

- **TrueGapless set** (every file from mpcenc 1.14 onward): the last frame's real length is in
  the header, and the sample count is **exact**.
- **TrueGapless clear**: the tail of the last frame is padding of unknown length and a fixed
  481-sample filterbank latency is removed instead. The result is an approximation — accurate
  to within one 1152-sample frame, i.e. under 30 ms.

Note the `last_frame_samples == 0 → 1152` normalisation happens *before* the branch, so in the
non-gapless case the subtraction is always exactly 481. `beg_silence` is 0 for SV7.

This is a **deviation from the original brief for this work**, which specified "samples ≈
frames × 1152 … exact=false for SV7". The reference reader is more precise than that, and the
extra two fields cost nothing to read, so `mpc.rs` implements the formula above and sets
`duration_exact` from `TrueGapless` — true for the modern-encoder case, false for the old one.

## Tags

APEv2, optionally followed by ID3v1, at the very end of the file — after the `SE` packet on
SV8. Measured and parsed by `tags/src/tail.rs`, shared with MP3, Monkey's Audio and WavPack.
