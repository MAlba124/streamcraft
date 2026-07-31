# Ogg Speex — header, comment packet and granule semantics

Reference material for `ogg/src/ident.rs`'s `parse_speex_head` and for `tags/src/ogg.rs`'s
`Format::OggSpeex` arm.

## Vendored, verbatim

- **`speex_header.h`** — the `SpeexHeader` struct declaration, i.e. the normative field list
  and order of the Ogg/Speex identification packet.
  Retrieved **2026-07-31** from
  `https://raw.githubusercontent.com/xiph/speex/master/include/speex/speex_header.h`.

**Licence status: redistributable.** "Copyright (C) 2002 Jean-Marc Valin", BSD-3-Clause
(Xiph.Org). The file is committed **unmodified, licence header intact**, which is what the
licence's source-redistribution condition asks for.

## Consulted but *not* vendered — and why

- ***The Speex Codec Manual*, Version 1.2 Beta 3, Jean-Marc Valin, December 8 2007** —
  §7.3 "Ogg file format" and Table 7.1 "Ogg/Speex header packet".
  Retrieved 2026-07-31 from `https://www.speex.org/docs/manual/speex-manual.pdf`.

  The manual is **not** BSD, contrary to what one might assume from the rest of Speex. Its
  title page reads:

  > Copyright © 2002-2007 Jean-Marc Valin/Xiph.org Foundation. Permission is granted to
  > copy, distribute and/or modify this document under the terms of the GNU Free
  > Documentation License, Version 1.1 or any later version published by the Free Software
  > Foundation; with no Invariant Section, with no Front-Cover Texts, and with no
  > Back-Cover.

  The GFDL does permit redistribution, but only of the *whole* document together with a copy
  of the licence — checking a 100-page codec manual and the GFDL text into `ogg/spec/` to
  document eighty bytes of header is not a sensible trade. The BSD `speex_header.h` above
  states the same field list normatively, so it is vendored instead and the manual is cited
  the way `mp4/spec/NOTES.md` cites the ISO standards. Short quotations below are
  attributed.

Clean-room: written from the two documents above plus the measurements recorded at the end.
No tag-library source was consulted.

## The identification packet (bos, packet 0)

Speex manual §7.3: "the first packet of the Ogg file contains the Speex header described in
table 7.1. All integer fields in the headers are stored as little-endian. The
`speex_string` field must contain the 'Speex   ' (with 3 trailing spaces), which identifies
the bit-stream. … The header packet has packetno=0 and granulepos=0."

**80 bytes**, `8 + 20 + 13 × 4`. Every integer is a signed 32-bit **little-endian** value
(`spx_int32_t`).

| offset | size | field                    | semantics                                          |
|--------|------|--------------------------|----------------------------------------------------|
| 0      | 8    | `speex_string`           | `"Speex   "` — 5 letters and **three** spaces       |
| 8      | 20   | `speex_version`          | encoder version string, NUL-padded, not terminated-guaranteed |
| 28     | 4    | `speex_version_id`       | header-format version; 1 for every released Speex   |
| 32     | 4    | `header_size`            | `sizeof(SpeexHeader)` — 80 for every released Speex |
| 36     | 4    | `rate`                   | **sampling rate in Hz**, and the unit of the granule |
| 40     | 4    | `mode`                   | 0 = narrowband, 1 = wideband, 2 = ultra-wideband     |
| 44     | 4    | `mode_bitstream_version` | bit-stream version ID of that mode                  |
| 48     | 4    | `nb_channels`            | **channel count**                                   |
| 52     | 4    | `bitrate`                | bit rate used, or −1 when not stated                |
| 56     | 4    | `frame_size`             | samples per Speex frame (160 nb / 320 wb / 640 uwb) |
| 60     | 4    | `vbr`                    | 1 = VBR, 0 = CBR                                    |
| 64     | 4    | `frames_per_packet`      | Speex frames packed into one Ogg packet             |
| 68     | 4    | `extra_headers`          | additional header packets **after the comments**    |
| 72     | 4    | `reserved1`              | "must be zero"                                      |
| 76     | 4    | `reserved2`              | "must be zero"                                      |

The magic `"Speex   "` is the only sniffing signal: unlike Vorbis and FLAC there is no
packet-type octet, and unlike Opus the magic is not a distinct word — the three trailing
spaces are load-bearing and must be matched.

`rate` and `nb_channels` are the only two fields a metadata view needs; `parse_speex_head`
returns exactly those. The rest steer decoding. Fields are validated the way the sibling
parsers validate theirs: a zero rate or zero channel count means an undecodable stream and
yields `None` rather than a nonsense property.

## The comment packet (packet 1)

Speex manual §7.3: "The second packet contains the Speex comment header. The format used is
the Vorbis comment format described here:
`http://www.xiph.org/ogg/vorbis/doc/v-comment.html`. This packet has packetno=1 and
granulepos=0."

Two consequences, both different from the sibling mappings:

- **No magic and no packet-type prefix.** Vorbis prefixes `0x03 "vorbis"`, Opus prefixes
  `"OpusTags"`; Speex prefixes *nothing*. The packet begins directly with the
  vendor-string length. So the raw comment body is handed to
  `pf_ogg::comment::parse_comment_body` rather than to one of the wrappers.
- **No framing bit.** Vorbis I §5.2.2.1 requires a trailing framing bit on its comment
  header because Vorbis headers are bit-packed; the Speex mapping inherits only the
  *comment format*, not that Vorbis-specific terminator. `parse_comment_body` does not
  require one, so nothing special is needed — but a parser that stripped a framing bit
  would eat a byte of the last comment.

## `extra_headers`

`extra_headers` counts header packets **after** the comment packet, before the first audio
packet. Every released Speex encoder writes 0. It is documented here because it is the one
field that could move the audio packets, and it explicitly cannot move the *comment* packet:
the comment is packet 1 unconditionally, so a tag scan that stops after two packets is
correct for any value of `extra_headers`. Only a decoder walking to the first audio packet
would need to skip them.

## Granule position — measured, not assumed

Speex manual §7.3: "The third and subsequent packets each contain one or more (number found
in header) Speex frames. These are identified with packetno starting from 2 and the
**granulepos is the number of the last sample encoded in that packet**."

So the plain reading is `duration = last_granule / rate`, which is the same arithmetic the
Vorbis and FLAC mappings use — `pf_ogg::duration::vorbis_duration_ns` is reused unchanged.

**What real files do.** Encoding exact-length sine sources with ffmpeg's `libspeex` encoder
and the `spx` (Ogg Speex) muxer, then reading the last page's granule directly
(2026-07-31, ffmpeg in the workspace devshell):

| source          | true samples | last granule | granule ÷ rate | `ffprobe` duration | delta   |
|-----------------|--------------|--------------|----------------|--------------------|---------|
| 8 kHz, 3.0 s    | 24000        | 23960        | 2.995000       | 3.000000           | 40 smp  |
| 8 kHz, 2.5 s    | 20000        | 19960        | 2.495000       | 2.500000           | 40 smp  |
| 16 kHz, 3.0 s   | 48000        | 47857        | 2.991063       | 3.000000           | 143 smp |
| 16 kHz, 10.0 s  | 160000       | 159857       | 9.991063       | 10.000000          | 143 smp |
| 32 kHz, 1.0 s   | 32000        | 31651        | 0.989094       | 1.000000           | 349 smp |

The granule is short by a **constant per mode**: 40 samples narrowband (5.0 ms), 143
wideband (8.9 ms), 349 ultra-wideband (10.9 ms). That constant is the encoder's
**algorithmic lookahead**: the Speex convention is that the encoder subtracts its lookahead
from the granule so the granule counts *content* samples, the decoder discarding that many
priming samples at the start. `ffprobe` adds the lookahead back and reports the nominal
input length.

**This parser does not add it back**, deliberately:

- The lookahead is **not in the file**. Opus can subtract its priming because RFC 7845 puts
  `pre_skip` in the identification header; Speex has no such field. Recovering it means a
  hard-coded mode→lookahead table taken from libspeex internals — a number with no normative
  source, which the workspace's citation rule would have nothing to point at.
- Whether the encoder subtracted it at all is an **encoder convention**, unreadable from the
  stream. Adding a constant back would silently over-report for any writer that did not.
- The error is bounded by 11 ms and is *always* an under-report, which is the safe direction
  for a library scan.

So `duration_exact` stays **true** — the granule is an authoritative sample count, in the
same sense as a Vorbis one — and the ≤11 ms shortfall against `ffprobe` is a documented,
bounded, spec-faithful difference rather than an estimate.
