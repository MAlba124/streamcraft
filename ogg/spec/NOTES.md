# sc-ogg spec notes

Interpretation decisions and errata for the Ogg implementation, per the plugin
conventions (spec: First-party codecs — "Spec errata and interpretation decisions get
documented in `spec/NOTES.md` per crate"). The normative text is `rfc3533.txt`
(RFC 3533, "The Ogg Encapsulation Format Version 0", May 2003). Code cross-references
sections as `§6` etc.

## Reference file

`rfc3533.txt` is the authoritative IETF text
(`https://www.rfc-editor.org/rfc/rfc3533.txt`). RFC 3533 is *informational* and, unlike
the FLAC RFC, contains no worked byte-level examples and — importantly — does not fully
specify the CRC parameters. The reference implementation (libogg) fixes them; see below.

## The CRC-32 (§6, field 7) — the one subtle spot

RFC 3533 says only: "a 32 bit CRC checksum of the page … The generator polynomial is
0x04c11db7." It does **not** state the initial value, bit reflection, or final XOR. The
`crc` module uses the parameters fixed by libogg, which are unusual:

- polynomial `0x04C11DB7`,
- initial value **0**,
- **no** input reflection, **no** output reflection (MSB-first),
- **no** final XOR.

This is the "direct"/non-reflected CRC-32 — same parameters as CRC-32/MPEG-2 but with
init 0 rather than all-ones. The reflected CRC-32 used by zlib/PNG/gzip would give a
different value and is *wrong* here.

Three layers of test pin this:
1. `crc::tests::known_answer_ascii_123456789` — the canonical `"123456789"` check value
   for these exact parameters is `0x89A1897F` (a reflected or init-all-ones CRC differs).
2. `crc::tests::table_step_matches_bitwise` — the 256-entry table equals the textbook
   bit-at-a-time division for every top-byte/input-byte pair.
3. `page::tests::real_libogg_vorbis_bos_page_crc_and_fields` — a **real bos page captured
   from `oggenc`** (libVorbis/libogg): our CRC recomputes the exact `0x453c7fce` that
   libogg stamped. `tests/cross_validate.rs` re-derives this live when `oggenc` is on
   `PATH` by walking every page of a freshly-encoded file.

## Byte order (§6)

"Fields with more than one byte length are encoded LSB (least significant byte) first" —
i.e. **little-endian**. So `granule_position` (u64), `bitstream_serial_number` (u32),
`page_sequence_number` (u32), and the `CRC_checksum` field itself are all little-endian
on the wire (`from_le_bytes` / `to_le_bytes`). The CRC *computation* is still MSB-first
over the byte stream; only the stored 4-byte field is little-endian.

## Segmentation and lacing (§5)

- A packet of length `L` → `L / 255` lacing values of 255, then one final value of
  `L % 255`. When `L` is a multiple of 255 the final value is **0** (§5: "A packet of
  255 bytes (or a multiple of 255 bytes) is terminated by a lacing value of 0"). A nil
  (zero-length) packet is a single lacing value of 0.
- A page holds at most 255 lacing values (`page_segments` is one byte, §6). A packet
  whose lacing values overflow the current page is split across pages; the continuation
  page sets the CONTINUED flag (§6, flag 0x01). The writer flushes a full segment table
  automatically; the reader rejoins across the flag.
- Max page: 255 segments × 255 bytes payload + 27-byte header + 255-byte table = 65307
  bytes (§6: "maximum 65307 bytes"), which our `MAX_PAGE_SIZE` matches.

## Granule position (§6, field 4)

A page's granule is that of the last packet to **finish** on it. The writer takes a
granule per *packet* and stamps it on whatever page the packet's terminating segment
lands on. A page on which no packet finishes (a packet still spilling into the next page)
gets the special value −1 == `u64::MAX` (`GRANULE_NONE`, §6: "A special value of -1 …
indicates that no packets finish on this page"). The demuxer surfaces each packet with
the granule of its finishing page verbatim — Ogg has "no concept of 'time'" (§4), so the
meaning is left to the media mapping, unaltered here.

## Decisions

- **Nil eos page on an empty/flushed stream.** `OggWriter::finish` guarantees a page with
  the eos flag. If a page is pending it becomes the eos page; otherwise a nil eos page
  (one zero-length segment) is emitted, which §6 explicitly allows ("Eos pages may be
  'nil' pages … containing no content but simply a page header with position information
  and the eos flag set"). Consequently, muxing *zero* packets still yields a valid stream
  of one bos+eos page, and the demuxer reads it back as one zero-length packet — it
  cannot know the muxer "meant nothing". Callers that flush every packet and then
  `finish` therefore get a trailing empty eos packet; callers that let `finish` flush the
  last packet's page get eos on that last data packet. Both are well-formed.

- **eos tagging on the reader.** A `Packet` is tagged `eos` only when it completes on the
  eos page's **last** segment — the stream's genuine final packet. A mid-page packet that
  happens to finish earlier on an eos page is not "the" eos packet. For the common
  one-packet-per-page case this is exactly right.

- **`bos` tagging.** The first packet emitted for a serial after its bos page is seen is
  tagged `bos`. By Ogg convention the bos page carries exactly the codec identification
  packet, so this is the id packet.

- **Resync / robustness (§6).** The capture pattern exists so a decoder can "regain
  synchronisation after parsing a corrupted stream". On any bad capture pattern, non-zero
  version, or CRC mismatch, the reader advances one byte and scans forward to the next
  `OggS`. It **never panics** on malformed input (the decoder-safety P0). A partial page
  at the tail of the currently-buffered bytes is held (streaming) or reported as leftover
  by `finish`. A lost page mid-packet orphans the partial packet: the fragment is dropped
  (rather than spliced across the gap and corrupting the packet), which is detectable via
  the gap in `page_sequence_number`.

- **Multiplexing.** Concurrent multiplexing ("grouping", §4) is supported on read: the
  demuxer keys reassembly by `bitstream_serial_number`, so interleaved logical streams
  come out as independent packet streams. The writer is single-serial by design; a caller
  muxing several streams runs one `OggWriter` per serial and interleaves whole pages
  (§4 requires all bos pages before any data page). Sequential multiplexing ("chaining",
  §4) — concatenated complete streams — is read transparently, since each page is
  self-describing by serial.

## Not yet / follow-ups

- **streamcraft elements** (`OggMux` / `OggDemux`). A demuxer needs one dynamic src pad
  *per discovered logical stream* and a muxer one dynamic sink pad per input; the
  milestone-1 Element pad model is static (`PadDesc { dynamic: false }`) and the scheduler
  has no per-stream pad add/remove. As `sc-flac` shipped the codec core before element
  polish, the tested reader/writer **library** is the deliverable; elements land when the
  core grows dynamic pads. The library depends on `streamcraft-core` only nominally and
  drops into an element wrapper unchanged.
- **Skeleton / chained-stream seeking metadata**, and granule→time conversion, are the
  concern of the media mapping / a higher layer, not the container.
- **`crc` SIMD / slice-by-N**: the byte-wise table CRC is already several hundred MB/s;
  a slice-by-8 variant would raise it further if the CRC ever dominates a profile.
