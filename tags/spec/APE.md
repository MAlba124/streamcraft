# Monkey's Audio (`.ape`) — field reference

Reference material for `tags/src/ape.rs`.

## Vendored, verbatim

- **`MonkeysAudio-APEHeader.h`** — `APE_COMMON_HEADER`, `APE_HEADER_OLD`, and the
  `CAPEHeader` interface. The normative declaration of the pre-3.98 header.
- **`MonkeysAudio-MACLib.h`** — `APE_DESCRIPTOR` and `APE_HEADER`, the two structures a
  version ≥ 3980 file opens with, plus the `APE_FORMAT_FLAG_*` values and the compression
  levels.
- **`MonkeysAudio-LICENSE.txt`** — the SDK licence.

All three retrieved **2026-07-31** from the Monkey's Audio SDK (MAC 12.13) at
`https://raw.githubusercontent.com/sbooth/CXXMonkeysAudio/main/{Source/MACLib/APEHeader.h,
Source/Shared/MACLib.h,LICENSE.txt}`, unmodified.

**Licence status: redistributable.** `LICENSE.txt` opens "Monkey's Audio License Agreement
(3 clause BSD) — Copyright 2000-2026 Matthew T. Ashland. All rights reserved." and then
states the literal three-clause BSD conditions, the first of which is that source
redistributions retain the notice — which is why the licence file is committed beside the
headers.

Cross-checked against the older MAC 4.33 SDK (`https://github.com/gchudov/MAC_SDK`,
`Source/MACLib/APEHeader.{h,cpp}`, `Source/MACLib/APEInfo.cpp`) for the pre-3.98 derivation
rules; the struct text is identical between the two generations. That mirror carries no
in-repo licence file (only a URL reference), so the current SDK's text is the one committed.

Clean-room: written from the headers above and the derivation code in the same SDK. No tag
library was consulted.

## Byte order and units

**Little-endian** throughout. `char cID[4]` is raw ASCII. Structures are `#pragma pack(4)`
in `MACLib.h`; explicit offsets are given below so packing is not something a parser has to
reason about.

**"Blocks" are frames.** One block is one sample point per channel — the same unit AIFF
calls a sample frame. `nBlocksPerFrame` therefore means "blocks in one APE *compression*
frame", and `nTotalBlocks / nSampleRate` is the duration in seconds.

## Locating the header

`CAPEHeader::FindDescriptor()`:

1. If the file begins with an ID3v2 tag, skip it: the synchsafe size from octets 6..9, plus
   10 for the header, plus another 10 when the footer flag (`0x10` in the ID3v2 flags octet)
   is set. Then skip any run of trailing zero padding.
2. Byte-scan forward for the magic, up to 1 MB.

The number of octets skipped is `nJunkHeaderBytes`, and **every offset below is relative to
it**.

**Magic**: `'MAC '` = `4D 41 43 20`. MAC 12.x additionally accepts `'MACF'` =
`4D 41 43 46` for large files. Both are the same layout.

```
  off  size  field       notes
  0    4     cID[4]      'MAC ' or 'MACF'
  4    2     nVersion    version × 1000 — 3.99 encodes as 3990
```

**Version dispatch**: `nVersion >= 3980` → the descriptor + header pair below;
`nVersion < 3980` → `APE_HEADER_OLD`. The *file format* version has been frozen at 3990
since MAC 4.06 — `APE_FILE_VERSION_NUMBER` is still 3990 in MAC 12.13, whose *application*
version is 12.13 — so essentially every file made this century takes the first path, and the
old path exists for files from 1999–2003 that are still in circulation.

## Version ≥ 3980

File layout:

```
  JUNK | APE_DESCRIPTOR | APE_HEADER | SEEK TABLE | HEADER DATA | APE FRAMES | TERMINATING DATA | TAG
```

### `APE_DESCRIPTOR` — 52 bytes

| off | size | field                    | semantics                                            |
|-----|------|--------------------------|------------------------------------------------------|
| 0   | 4    | `cID[4]`                 | `'MAC '` / `'MACF'`                                   |
| 4   | 2    | `nVersion`               | version × 1000                                        |
| 6   | 2    | `nPadding`               | "because 4-byte alignment requires this (or else nVersion would take 4-bytes)". Value is meaningless — but the **two octets are real** and a parser that omits them mis-reads every field after |
| 8   | 4    | `nDescriptorBytes`       | total descriptor size. **Seek this far from the descriptor start to reach `APE_HEADER`** — do not hard-code 52 |
| 12  | 4    | `nHeaderBytes`           | size of the `APE_HEADER` that follows — do not hard-code 24 |
| 16  | 4    | `nSeekTableBytes`        | `nSeekTableElements = nSeekTableBytes / 4`            |
| 20  | 4    | `nHeaderDataBytes`       | stored prefix of the original file (its WAV/AIFF header) |
| 24  | 4    | `nAPEFrameDataBytes`     | compressed audio size, low 32 bits                    |
| 28  | 4    | `nAPEFrameDataBytesHigh` | …high 32 bits                                         |
| 32  | 4    | `nTerminatingDataBytes`  | post-audio bytes of the original file, tag excluded    |
| 36  | 16   | `cFileMD5[16]`           | MD5 of the file (computed out of order — see the SDK note) |

The two size fields at offsets 8 and 12 are the forward-compatibility mechanism: a later SDK
may grow either structure, and a reader that hops by the declared sizes keeps working. This
parser hops.

### `APE_HEADER` — 24 bytes, at `descriptor + nDescriptorBytes`

| off | size | field               | semantics                                              |
|-----|------|---------------------|--------------------------------------------------------|
| 0   | 2    | `nCompressionLevel` | 1000 fast, 2000 normal, 3000 high, 4000 extra high, 5000 insane |
| 2   | 2    | `nFormatFlags`      | see the flag table below                               |
| 4   | 4    | `nBlocksPerFrame`   | **explicit** here (derived in the old header)          |
| 8   | 4    | `nFinalFrameBlocks` | blocks in the last frame                               |
| 12  | 4    | `nTotalFrames`      | frame count                                            |
| 16  | 2    | `nBitsPerSample`    | **explicit** here (derived in the old header)          |
| 18  | 2    | `nChannels`         | 1 … 32                                                 |
| 20  | 4    | `nSampleRate`       | Hz                                                     |

Sanity limits the SDK itself applies, and which this parser mirrors so a corrupt header
cannot become a plausible-looking duration: `nBlocksPerFrame > 0`; `nBlocksPerFrame` above
1 000 000 is invalid unless `nCompressionLevel >= 5000`, where the ceiling is 10 000 000;
`nFinalFrameBlocks > nBlocksPerFrame` is invalid; `nChannels` must be 1…32.

## Version < 3980 — `APE_HEADER_OLD`, 32 bytes

| off | size | field               |
|-----|------|---------------------|
| 0   | 4    | `cID[4]` = `'MAC '` |
| 4   | 2    | `nVersion`          |
| 6   | 2    | `nCompressionLevel` |
| 8   | 2    | `nFormatFlags`      |
| 10  | 2    | `nChannels`         |
| 12  | 4    | `nSampleRate`       |
| 16  | 4    | `nHeaderBytes`      |
| 20  | 4    | `nTerminatingBytes` |
| 24  | 4    | `nTotalFrames`      |
| 28  | 4    | `nFinalFrameBlocks` |

Note what is *missing*: `nBlocksPerFrame` and `nBitsPerSample`. Both are **derived**:

```c
// blocks per frame
nBlocksPerFrame = ((nVersion >= 3900) ||
                   ((nVersion >= 3800) && (nCompressionLevel == 4000 /* extra high */)))
                  ? 73728 : 9216;
if (nVersion >= 3950) nBlocksPerFrame = 73728 * 4;   // 294912

// bits per sample
nBitsPerSample = (nFormatFlags & 1 /* 8_BIT */)  ? 8
               : (nFormatFlags & 8 /* 24_BIT */) ? 24 : 16;
```

so three tiers: **9216** below 3800 (and 3800–3899 at any level but extra-high), **73728**
at 3900+ or 3800+ extra-high, **294912** at 3950+. The second assignment is unconditional
for ≥ 3950 and overrides the first.

Fields that follow the 32-byte struct, **in this order** and only when their condition holds:

1. `nFormatFlags & 4` (`HAS_PEAK_LEVEL`) → `uint32 nPeakLevel` at offset 32.
2. `nFormatFlags & 16` (`HAS_SEEK_ELEMENTS`) → `uint32 nSeekTableElements`, after the peak
   level when that is present. Otherwise `nSeekTableElements = nTotalFrames`.
3. `!(nFormatFlags & 32)` (`CREATE_WAV_HEADER` **clear**) → `nHeaderBytes` octets of the
   stored original WAV header.
4. the seek table: `uint32[nSeekTableElements]`.
5. `nVersion <= 3800` → a seek *bit* table: `uint8[nSeekTableElements]`.

Note the ordering inversion against the modern layout: old files store the **WAV header
before** the seek table, new files store the **seek table before** the header data. Neither
matters to a tag scan — nothing past the header is read — but getting it wrong would matter
to anything that did.

### Format flags

| bit | value | name                | meaning                                              |
|-----|-------|---------------------|------------------------------------------------------|
| 0   | 1     | `8_BIT`             | 8-bit samples [obsolete]                              |
| 1   | 2     | `CRC`               | newer CRC32 error detection [obsolete]                |
| 2   | 4     | `HAS_PEAK_LEVEL`    | a `uint32` peak level follows the header [obsolete]   |
| 3   | 8     | `24_BIT`            | 24-bit samples [obsolete]                             |
| 4   | 16    | `HAS_SEEK_ELEMENTS` | a `uint32` seek-element count follows                 |
| 5   | 32    | `CREATE_WAV_HEADER` | synthesise the WAV header on decode — i.e. **none is stored** |
| 6   | 64    | `AIFF`              | source was AIFF                                       |
| 7   | 128   | `W64`               | source was Sony Wave64                                |
| 8   | 256   | `SND`               | source was SND                                        |
| 9   | 512   | `BIG_ENDIAN`        | big-endian sample encoding                            |
| 10  | 1024  | `CAF`               | source was CAF                                        |
| 11  | 2048  | `SIGNED_8_BIT`      | 8-bit values are signed                               |
| 12  | 4096  | `FLOATING_POINT`    | floating-point samples                                |

Bits 6–12 exist only in the modern SDK and appear only on ≥ 3980 files. Only bits 0, 3, 4, 5
affect the pre-3.98 parse; none of them affects the duration.

## Total blocks, and the `nTotalFrames == 0` trap

Identical in both code paths and both SDK generations:

```c
nTotalBlocks = (nTotalFrames == 0) ? 0
             : (int64)(nTotalFrames - 1) * (int64)nBlocksPerFrame + (int64)nFinalFrameBlocks;
```

The zero guard is not decoration: without it `nTotalFrames - 1` underflows to `0xFFFFFFFF`
and the file reports roughly 2.8 million years. A hostile or merely unfinalised `.ape` is
exactly the input that hits it, so the guard is reproduced verbatim in `ape.rs` and pinned by
a test.

The SDK goes further on the **old** path only: `if (nTotalFrames == 0) return
ERROR_UNDEFINED;` — "fail on 0 length APE files (catches non-finalized APE files)". The
modern path tolerates it and yields zero blocks.

### Duration

```
  duration = nTotalBlocks / nSampleRate
```

**Exact**: `nTotalBlocks` is a declared block count, not a size divided by a bitrate.

## Tags

APEv2 — the tag format Monkey's Audio originated, and which WavPack and Musepack then
adopted — optionally followed by ID3v1, both at the very end of the file. Measured and parsed
by `tags/src/tail.rs`, shared with MP3, WavPack and Musepack.
