# AIFF / AIFF-C — field reference

Reference material for `tags/src/aiff.rs`. Written in the `mp4/spec/NOTES.md` style: the
normative documents are **Apple's**, and Apple's papers carry no redistribution grant, so
they are **not vendored**. This file instead records every field, offset, width, endianness
and semantic the parser relies on — completely enough that a future clean-room
implementation needs no network access.

## Origin

- ***Audio Interchange File Format: "AIFF"*, Apple Computer, Inc., Version 1.3,
  January 4 1989.** The base format: the `FORM`/`AIFF` container, the Common Chunk, the
  Sound Data Chunk, and the four "EA IFF 85" text chunks.
  Consulted at `https://mmsp.ece.mcgill.ca/Documents/AudioFormats/AIFF/Docs/AIFF-1.3.pdf`
  (McGill MMSP's mirror of the Apple paper), retrieved **2026-07-31**.
- ***Audio Interchange File Format AIFF-C: "AIFF-C"*, Apple Computer, Inc.,
  Draft 3-24-89 / revised August 26 1991.** The compressed variant: form type `AIFC`, the
  Common Chunk's two extra fields, and the `FVER` chunk.
  Consulted at `https://web.archive.org/web/20071219035740/http://www.cnpbagwell.com/aiff-c.txt`,
  retrieved **2026-07-31**.
- **"EA IFF 85" Standard for Interchange Format Files**, Electronic Arts, 1985 — the
  chunk grammar AIFF inherits (cited by the Apple paper rather than reproduced here).
- **`id3v2.4.0-structure` / `id3v2.3.0`** for the payload of the `ID3 ` chunk. Parsing is
  `pf_mp3::id3`'s, not this crate's.

Clean-room: written from the documents above only. No tag-library source was consulted.

## Byte order

**Big-endian, everywhere.** AIFF is a 1988 Macintosh (Motorola 68000) format; every
multi-byte field in the container — `ckSize`, `numChannels`, `numSampleFrames`,
`sampleSize`, and the 80-bit `sampleRate` — is most-significant byte first. This is the one
structural difference from RIFF/WAVE, which is otherwise the same chunk grammar
little-endian. Getting it backwards yields a plausible-looking but absurd duration, not a
parse failure, so it is worth stating loudly.

## Chunk grammar (AIFF-1.3, "File Structure")

```
  Chunk:  ID   ckID     4 bytes, ASCII
          long ckSize   4 bytes, big-endian, SIGNED in the paper's C declaration
          char ckData[] ckSize bytes
```

- `ckSize` **excludes** the 8 header bytes: "ckSize is the size of the data portion of the
  chunk, in bytes. It does not include the 8 bytes used by ckID and ckSize."
- **Word alignment.** "If the data is an odd number of bytes in length, a zero pad byte must
  be added at the end. The pad byte is not included in ckSize." So the stride from one chunk
  to the next is `8 + ckSize + (ckSize & 1)` — identical to RIFF.
- `ckSize` is declared `long` (signed) in Apple's C. A file claiming a negative size is
  malformed; the parser reads it unsigned and lets the bounds checks reject it.

## The FORM container

```
  offset  size  field     value
  0       4     ckID      'FORM'  (0x464F524D)
  4       4     ckSize    big-endian; size of formType + chunks[]
  8       4     formType  'AIFF' (0x41494646) or 'AIFC' (0x41494643)
  12      …     chunks[]
```

`formType` is `AIFF` for the 1989 base format and `AIFC` for the 1991 compressed variant.
Both are walked identically here: the chunk grammar and every chunk this scanner reads are
unchanged between them.

**Sniff**: `FORM` at offset 0 **and** `AIFF`/`AIFC` at offset 8. The form type is required —
`FORM` alone also introduces IFF-85 documents that are not audio at all (`8SVX`, `ILBM`,
`ANIM`).

## Common Chunk `COMM` (AIFF-1.3, "Common Chunk")

Required; exactly one per FORM.

```
  rel.off  size  type      field            semantics
  0        2     short     numChannels      1 = mono, 2 = stereo, … ; big-endian
  2        4     ulong     numSampleFrames  count of sample FRAMES (not bytes, not points)
  6        2     short     sampleSize       bits per sample point, 1..32
  8        10    extended  sampleRate       80-bit IEEE 754 extended, frames per second
  ---- AIFF-C (form type 'AIFC') only, ckSize >= 22: ----
  18       4     ID        compressionType  e.g. 'NONE', 'sowt', 'fl32', 'ima4', 'ulaw'
  22       …     pstring   compressionName  Pascal string, human-readable
```

- `ckSize` is **18** for AIFF and **≥ 22** for AIFF-C. A parser must therefore not require
  22 bytes before reading the first 18.
- `numSampleFrames`: "the number of sample frames, not the number of bytes nor the number of
  sample points". One frame holds `numChannels` sample points. **This is a frame count
  regardless of compression** — an AIFF-C `ima4` or `ulaw` file states the number of decoded
  frames here just as a `NONE` file does, which is what makes the duration below exact for
  both.
- `sampleSize` "can be any number from 1 to 32". Reported as-is; a `sampleSize` of 0 is
  malformed and simply reported as nothing.

### Duration

```
  duration = numSampleFrames / sampleRate
```

**Exact** — `numSampleFrames` is a decoded-frame count declared by the file rather than a size
divided by a bitrate. This is the AIFF equivalent of FLAC's STREAMINFO sample count, not of
MP3's CBR estimate. A `sampleRate` that does not parse to a positive integer (see below)
yields no duration rather than a wrong one.

`numSampleFrames == 0` is legal and means an empty sound (AIFF-1.3 allows the `SSND` chunk
to be absent entirely in that case); it yields a duration of zero, not `None`.

#### …but only when the compression type says one unit is one frame

Both papers are unambiguous that the field counts sample frames, with no exception for
compressed data: "numSampleFrames is the number of sample frames, not the number of bytes nor
the number of sample points in the Sound Data Chunk."

**Real files disagree.** Measured 2026-07-31 on an AIFF-C written by ffmpeg
(`ffmpeg -i src.wav -c:a adpcm_ima_qt out.aifc`, 3.000 s of 44.1 kHz stereo):

| field                     | value                                    |
|---------------------------|------------------------------------------|
| `FORM` type               | `AIFC`                                   |
| `COMM` `ckSize`           | 24                                       |
| `compressionType`         | `ima4`                                   |
| `numSampleFrames`         | **2068**                                 |
| `sampleRate`              | 44100                                    |
| true length (`ffprobe`)   | 3.001179 s = **132352 frames**           |

`132352 / 2068 = 64`. QuickTime's `ima4` packs **64 sample frames per packet per channel**,
and the convention — which ffmpeg follows and QuickTime originated — is that `numSampleFrames`
holds the *packet* count for that codec. Taking the field literally reports **47 ms for a
3-second file**: wrong by 64x, and wrong in a way that looks like a plausible short sound
rather than like a parse failure.

There is no per-codec frames-per-packet table in either Apple paper, and inventing one from
recall would be exactly the kind of unsourced constant the workspace's citation rule exists to
prevent. So `aiff.rs` believes `numSampleFrames` **only** when the compression type means "one
unit is one sample frame":

| `compressionType` | source | meaning |
|---|---|---|
| *(absent — an 18-octet `COMM`)* | AIFF-1.3 | a plain AIFF; there is no compression field |
| `NONE` | AIFF-C, "Compression Type IDs" | "not compressed" |
| `twos`, `sowt` | QTFF sound four-CCs | 16-bit PCM, big- and little-endian |
| `raw ` | QTFF | 8-bit offset-binary PCM |
| `in24`, `42ni`, `in32`, `23ni` | QTFF | 24- and 32-bit integer PCM, both byte orders |
| `fl32`, `FL32`, `fl64`, `FL64` | QTFF | IEEE 32- and 64-bit float |

Anything else — `ima4`, `ulaw`, `alaw`, `MAC3`, `QDM2`, a codec four-CC this table does not
know — yields **no duration at all**. The sample rate and channel count are still reported:
those are unambiguous whatever the compression is, and a missing duration is honest where a
64x-wrong one is not.

## The 80-bit IEEE 754 extended sample rate

AIFF-1.3 defines the type as "80 bit IEEE Standard 754 floating point number (Standard
Apple Numeric Environment [SANE] data type Extended)". This is the x87 80-bit
double-extended format, **big-endian** on the wire:

```
  bit   79       78 … 64      63        62 … 0
        sign     exponent     integer   fraction
        (1)      (15 bits)    bit (1)   (63 bits)

  byte  0        0 … 1        2 … 9  (the 64-bit mantissa, integer bit included)
```

The exponent is **biased by 16383**. Unlike IEEE binary32/binary64, the leading integer bit
is **explicit** — it is bit 63 of the 64-bit mantissa field, not implied. Therefore:

| exponent `e`  | mantissa `m`      | value                                     |
|---------------|-------------------|-------------------------------------------|
| `0`           | `0`               | ±0                                        |
| `0`           | `!= 0`            | denormal: `±m × 2^(-16382-63)`            |
| `1 … 0x7FFE`  | any               | `±m × 2^(e - 16383 - 63)`                 |
| `0x7FFF`      | `0x8000…0`        | ±∞                                        |
| `0x7FFF`      | other             | NaN                                       |

Note the value formula is the same for normals and denormals once the mantissa is treated as
a plain 64-bit integer scaled by `2^-63`; only the exponent's effective value differs
(`e - 16383` vs. the fixed `-16382`). Because the integer bit is explicit, encodings with
`e != 0` and bit 63 clear ("unnormals") are representable but invalid; they are not special
cased — the general formula gives a small value, which then fails the "is this a plausible
sample rate" test.

**Worked encodings** (the four the unit tests pin):

| rate    | 10 bytes, big-endian                | `e`      | shift `e-16383-63` | `m >> 48` |
|---------|-------------------------------------|----------|--------------------|-----------|
| 44100   | `40 0E AC 44 00 00 00 00 00 00`     | `0x400E` | −48                | `0xAC44`  |
| 48000   | `40 0E BB 80 00 00 00 00 00 00`     | `0x400E` | −48                | `0xBB80`  |
| 22050   | `40 0D AC 44 00 00 00 00 00 00`     | `0x400D` | −49                | `0x5622`  |
| 8000    | `40 0B FA 00 00 00 00 00 00 00`     | `0x400B` | −51                | `0x1F40`  |

The parser evaluates `m × 2^shift` in integers (no `f64` round-trip): a negative shift is a
right shift with round-half-up, a non-negative shift means a value of at least `2^63`, which
is not a sample rate. Zero, negative, infinite, NaN and out-of-`u32` results all yield
"no rate", never a panic and never a wrong number.

## Text chunks (AIFF-1.3, "Text Chunks — Name, Author, Copyright, Annotation")

All four are inherited from EA IFF 85; each is optional and holds nothing but text.

| `ckID`  | hex           | meaning     | mapped tag  |
|---------|---------------|-------------|-------------|
| `NAME`  | `4E 41 4D 45` | Name        | `TITLE`     |
| `AUTH`  | `41 55 54 48` | Author      | `ARTIST`    |
| `(c) `  | `28 63 29 20` | Copyright   | `COPYRIGHT` |
| `ANNO`  | `41 4E 4E 4F` | Annotation  | `COMMENT`   |

- The copyright id is **lowercase `c`, and the fourth octet is a space** (0x20): "For the
  Copyright Chunk, the 'c' is lowercase and there is a space (0x20) after the close
  parenthesis." The chunk id itself "serves as the copyright characters '©'", so the text is
  the notice *without* a leading ©.
- `text` "contains pure ASCII characters. It is not a pstring nor a C string. The number of
  characters in text is determined by ckSize." So: **not** NUL-terminated, and not
  length-prefixed. A trailing NUL is nonetheless tolerated by the parser, because taggers
  write them.
- At most one `NAME`, `AUTH` and `(c) ` per FORM; **many** `ANNO` chunks may exist. The
  first is taken (the sink's `get` returns the first value for a key anyway).
- "pure ASCII" is the letter of the spec; files in the wild carry Latin-1 and UTF-8. The
  parser runs the same `decode_text` fallback the RIFF `INFO` reader uses: valid UTF-8 is
  passed through, anything else is transcoded from Latin-1.

## The `ID3 ` chunk

**Not in either Apple paper.** It is a de-facto convention: ffmpeg (`-write_id3v2 1`),
iTunes and several taggers store a complete ID3v2 tag as the body of a chunk whose id is
`ID3 ` (`49 44 33 20` — three characters and a trailing space). Some writers use the
lowercase `id3 ` variant; both are accepted.

The chunk body is a **complete ID3v2 tag starting at its `"ID3"` magic**, exactly as it
would appear at the front of an MP3, so it is handed to `pf_mp3::id3::parse_v2` unchanged.

### Precedence: ID3 wins

The sink returns the **first** value seen for a key, so emission order is precedence order.
This parser emits the `ID3 ` chunk's frames **before** the text chunks, in a separate pass
over the same window, regardless of which comes first in the file. Reasoning:

- ID3v2 is strictly richer: it has album, track number, date, genre and ReplayGain, where
  the IFF text chunks have only four fields.
- ID3v2 states its own text encoding per frame; the IFF text chunks are "pure ASCII" by
  spec and ambiguous in practice.
- A writer that emits both wrote the ID3 tag on purpose and the text chunks as a
  compatibility courtesy — an AIFF written by ffmpeg with `-metadata title=X
  -write_id3v2 1` carries `X` in both places, and where they differ the ID3 one is the one
  the user edited last.

This costs one extra walk of the chunk *headers* (no payload is touched twice), which is a
few dozen compares.

## Other chunks

`SSND` (Sound Data), `MARK`, `INST`, `COMT`, `APPL`, `FVER`, `MIDI`, `AESD` are **skipped by
size**. In particular `SSND` — the audio — is never read: the duration comes from `COMM`,
so a 500 MB AIFF costs the same one prefix read as a small one, and a text or `ID3 ` chunk
placed after it costs one positioned follow-up read at the offset the chunk walk reports.
