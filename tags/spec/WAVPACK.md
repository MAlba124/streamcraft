# WavPack — spec provenance and the two things the vendored document omits

Reference material for `tags/src/wv.rs`.

## Vendored, verbatim

- **`WavPack5FileFormat.pdf`** — *WavPack 4 & 5 Binary File / Block Format*, David Bryant,
  April 12 2020. The normative block-format document.
  Retrieved **2026-07-31** from
  `https://raw.githubusercontent.com/dbry/WavPack/master/doc/WavPack5FileFormat.pdf`
  (the `doc/` directory of the reference implementation's repository).
- **`WavPack5FileFormat.txt`** — a plain-text rendering of the same PDF, produced locally
  with `pdftotext -layout`. Committed for grep/review convenience; the PDF is the original.
  Nothing was edited.
- **`WavPack-COPYING.txt`** — the WavPack project's licence, retrieved the same day from
  `https://raw.githubusercontent.com/dbry/WavPack/master/COPYING`.

**Licence status: redistributable.** WavPack is BSD-3-Clause, "Copyright (c) 1998 - 2025
David Bryant, All rights reserved", and the licence permits redistribution in source form
provided the copyright notice, the conditions and the disclaimer are retained —
`WavPack-COPYING.txt` is committed alongside for exactly that reason. Section headings in
`src/wv.rs` cite this document as e.g. "(WavPack file format §2.0)".

Clean-room: the parser is written from the document above and the two reference-source facts
recorded below. No tag-library source was consulted.

## What the document does not say, and where the answers come from

The vendored PDF is complete about the *layout* but leaves two things to the reference
implementation. Both are recorded here so the parser needs nothing else.

### 1. The 15-entry standard sample-rate table

§2.0 says only "sampling rate (if one of 15 standard rates)" and that flags bits 26–23 hold
"sampling rate (1111 = unknown/custom)". The table itself lives in the reference source:

> `src/common_utils.c`, line 31 (WavPack master, retrieved 2026-07-31,
> `https://raw.githubusercontent.com/dbry/WavPack/master/src/common_utils.c`):
> ```c
> const uint32_t sample_rates [] = { 6000, 8000, 9600, 11025, 12000, 16000, 22050,
>     24000, 32000, 44100, 48000, 64000, 88200, 96000, 192000 };
> ```

| index | rate  | index | rate  | index | rate   |
|-------|-------|-------|-------|-------|--------|
| 0     | 6000  | 5     | 16000 | 10    | 48000  |
| 1     | 8000  | 6     | 22050 | 11    | 64000  |
| 2     | 9600  | 7     | 24000 | 12    | 88200  |
| 3     | 11025 | 8     | 32000 | 13    | 96000  |
| 4     | 12000 | 9     | 44100 | 14    | 192000 |
| —     | —     | —     | —     | **15**| **custom — see `ID_SAMPLE_RATE`** |

The masks confirming the field position, from `include/wavpack.h` (same retrieval):
`#define SRATE_LSB 23`, `#define SRATE_MASK (0xfL << SRATE_LSB)`, `#define MONO_FLAG 4`.

### 2. The `ID_SAMPLE_RATE` (0x27) sub-block payload

§3.0 lists the id but not its contents. From `src/open_utils.c`'s `read_sample_rate`
(same retrieval):

> the payload is **3 or 4 bytes, little-endian**; with 4 bytes the top byte is masked
> `0x7F`, i.e. the field is 31 bits. (The 4-byte form exists "for sampling rates >
> 16777215".)

`src/wv.rs` implements the sub-block walk for this, so a file with rate index 15 still
reports its rate and duration.

## The block header (§2.0) — 32 bytes, little-endian

```
  off  size  field             notes
  0    4     ckID              "wvpk"
  4    4     ckSize            size of the entire block minus 8
  8    2     version           0x402 … 0x410 valid for decode
  10   1     block_index_u8    upper 8 bits of the 40-bit block_index   (v5; 0 in v4)
  11   1     total_samples_u8  upper 8 bits of the 40-bit total_samples (v5; 0 in v4)
  12   4     total_samples     lower 32 bits; valid only when block_index == 0
  16   4     block_index       lower 32 bits of the first sample's index
  20   4     block_samples     samples in this block; 0 = a non-audio block
  24   4     flags             see below
  28   4     crc               crc of the decoded data
```

"Samples" here means **frames** — "a complete sample for all channels".

### The 40-bit fields, and "unknown length"

§2.0: "the 40-bit `total_samples` reserves values with the lower 32 bits all set to
represent an unknown length, so loading and storing this is a little tricky (see
`src/wavpack_local.h` for macros to do this)". The macros are in fact in
`include/wavpack.h` (retrieved 2026-07-31):

```c
#define GET_BLOCK_INDEX(hdr) ( (int64_t) (hdr).block_index + ((int64_t) (hdr).block_index_u8 << 32) )

#define GET_TOTAL_SAMPLES(hdr) ( ((hdr).total_samples == (uint32_t) -1) ? -1 : \
    (int64_t) (hdr).total_samples + ((int64_t) (hdr).total_samples_u8 << 32) - (hdr).total_samples_u8 )
```

So:

- `total_samples == 0xFFFFFFFF` (**regardless of the upper 8 bits**) means **unknown**, and
  the scanner reports no duration rather than a wrong one.
- otherwise the value is `lower + (upper << 32) - upper`. The `- upper` term is the
  reserved-value skip: each 2^32 block loses the all-ones encoding, so the mapping stays
  dense. For every file below 2^32 frames (≈27 hours at 44.1 kHz) `upper` is 0 and this
  degenerates to the plain lower 32 bits.

§2.0 also notes that a file made through a pipe may carry a lower-32 value of **1** as a
"length unknown" marker and expects the decoder to seek to the end. This scanner does not
do that seek — a duration derived from a fabricated 1-sample header would be worse than
none. Such a file reports 1 frame; it is rare enough (and identifiable) not to warrant a
second read.

### Flags (§2.0), the bits this parser reads

```
  bits 1,0    bytes/sample − 1
  bit  2      MONO_FLAG: 0 = stereo output, 1 = mono output
  bit  3      hybrid mode
  bits 22-18  maximum magnitude of decoded data (bits required − 1)
  bits 26-23  sampling rate index (1111 = unknown/custom)
  bit  30     FALSE_STEREO: block is stereo but the data is mono
  bit  31     DSD_FLAG: 1 = DSD audio (ver 5.0+)
```

**Channel count.** The header states only mono-or-stereo. A multichannel file is "divided
into some number of stereo and mono streams and multiplexed into separate blocks which
repeat in sequence" (§1.0), with `INITIAL_BLOCK` (bit 11) / `FINAL_BLOCK` (bit 12) marking
the sequence and the true count carried in an `ID_CHANNEL_INFO` (0x0D) sub-block. For a
mono or stereo file — which is what a tag scan overwhelmingly meets — both sequence bits are
set on every block and `MONO_FLAG` is the whole answer. `src/wv.rs` reads
`ID_CHANNEL_INFO` when it is present in the first audio block (its first byte is the channel
count), and falls back to `MONO_FLAG` otherwise.

### Which block

§1.0: "the first block that contains audio samples in a WavPack file determines the format of
the entire file". Blocks "may contain only metadata, especially at the beginning and end of a
file", so the walk skips blocks with `block_samples == 0` when looking for the format, and
takes `total_samples` from a block with `block_index == 0`.

## Metadata sub-blocks (§3.0)

Everything after the 32-byte header to the end of the block:

```
  uchar id     0x3F  unique metadata function id
               0x20  decoder needn't understand this metadata
               0x40  actual data byte length is one LESS than the word count implies
               0x80  large block (> 255 words)
  uchar ws     small block: data size in 16-bit WORDS
  uchar ws[3]  large block: data size in words, little-endian
  data[]       padded to an even number of bytes
```

So the payload byte length is `words * 2 - ((id & 0x40) != 0)`, and the sub-block's total
length is `2 + words * 2` (small) or `4 + words * 2` (large) — always even, which is what
keeps the walk aligned. `ws == 0` is legal (a sub-block that signals by its presence).

## Tags (§4.0)

> "Both the APEv2 tags and/or ID3v1 tags must come at the end of the WavPack file, with the
> ID3v1 coming last if both are present."

Which is byte-for-byte the layout `tags/src/tail.rs` already measures for MP3, Monkey's
Audio and Musepack. §4.0 lists the recommended APEv2 field names (`Artist`, `Title`,
`Album`, `Track`, `Year`, `Genre`, `Comment`, `Replaygain_*`, `Cover Art (Front)`, …) but
explicitly places "no restrictions on what field names may be used"; the key mapping is
`pf_mp3::id3::parse_ape`'s.
