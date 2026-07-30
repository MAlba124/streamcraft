# pf-mp4 — spec provenance and design notes

The ISO/IEC standards this crate is written against are **copyrighted and cannot be
vendored** (unlike the FLAC RFC 9639, the Matroska RFC 9559, or the Ogg RFC 3533 checked
into their crates' `spec/` directories). This file instead records the **exact editions**
each `§` citation in the source refers to, so a reviewer with access to the texts can
follow the code clause by clause, and so the citations do not silently drift between
standard revisions.

## Standards cited

- **ISO/IEC 14496-12** — *Information technology — Coding of audio-visual objects — Part
  12: ISO base media file format*. The box (atom) grammar: the `size`/`type` header, the
  `largesize`/`uuid` extensions (§4.2), FullBox version+flags (§4.2), and the container /
  leaf boxes the sample-table resolver reads. Citations in `src/boxes.rs` (e.g. `§8.7.4`
  for `stsc`) are to the **6th edition (2022)** clause numbering; the sample-table
  structure has been stable since the 2nd edition (2005), so the clause text is
  unambiguous even against an earlier printing. Sections used:
  - §4.2 Box / FullBox structure
  - §4.3 File Type Box (`ftyp`)
  - §8.1.1 Media Data Box (`mdat`)
  - §8.2.1/§8.2.2 Movie Box (`moov`) / Movie Header Box (`mvhd`)
  - §8.3.1/§8.3.2 Track Box (`trak`) / Track Header Box (`tkhd`)
  - §8.4.1–§8.4.4 Media Box (`mdia`) / Media Header (`mdhd`) / Handler (`hdlr`) /
    Media Information Box (`minf`)
  - §8.5.1/§8.5.2 Sample Table Box (`stbl`) / Sample Description Box (`stsd`)
  - §8.6.1.2 Decoding Time to Sample Box (`stts`)
  - §8.6.1.3 Composition Time to Sample Box (`ctts`) — v0 unsigned, v1 signed offsets
  - §8.6.2 Sync Sample Box (`stss`)
  - §8.6.5/§8.6.6 Edit Box (`edts`) / Edit List Box (`elst`)
  - §8.7.3.2/§8.7.3.3 Sample Size Box (`stsz`) / Compact Sample Size Box (`stz2`)
  - §8.7.4 Sample To Chunk Box (`stsc`)
  - §8.7.5 Chunk Offset Boxes (`stco` 32-bit, `co64` 64-bit)
  - §8.8.1/§8.8.4 Movie Extends Box (`mvex`) / Movie Fragment Box (`moof`) — DETECTED so
    the demuxer errors loudly; fragmented MP4 is out of v1 scope.
  - §12.1.3 VisualSampleEntry, §12.2.3 AudioSampleEntry — the `stsd` entry layouts.

- **ISO/IEC 14496-15** — *… Part 15: Carriage of network abstraction layer (NAL) unit
  structured video in the ISO base media file format*. The `AVCDecoderConfigurationRecord`
  (§5.3.3.1) and `HEVCDecoderConfigurationRecord` (§8.3.3.1) carried in the `avcC` / `hvcC`
  child of an `avc1` / `hvc1` sample entry. **This crate does not re-parse these records**:
  the identical record rides an MKV `CodecPrivate`, so `src/codec.rs` reuses `pf-mkv`'s
  already-tested `nal_head_from_config` (see "Reframer reuse" below). Edition: 2022.

- **RFC 6381** — *The 'Codecs' and 'Profiles' Parameters for "Bucket" Media Types*. The
  four-CC ⇒ codec family names in `src/codec.rs::family_for` follow the sample-entry
  four-CCs RFC 6381 §3.3 registers (`avc1`/`avc3`, `hvc1`/`hev1`, `vp09`, `av01`, `Opus`,
  `mp4a`). Not vendored (RFC text is freely available at rfc-editor.org).

## Reframer reuse (why pf-mp4 depends on pf-mkv)

`avc1`/`hvc1` sample entries store frames as **length-prefixed NAL units** and their
parameter sets in an `avcC`/`hvcC` configuration record — byte-for-byte the same record
Matroska carries in `CodecPrivate`, and the same length-prefixed → Annex B reframing MKV
Blocks need. Rather than copy `pf-mkv`'s tested `parse_avcc`/`parse_hvcc` +
`length_prefixed_to_annex_b`, pf-mp4 depends on pf-mkv and reuses
`pf_mkv::nal_head_from_config` + `pf_mkv::Reframer`. One implementation, tested once. This
is **not** a container dependency: the MP4 box grammar, sample-table resolution, and
streaming slicer are all hand-written in this crate; only the codec-record parser (a
14496-15 concern, not a 14496-12 one) is shared.

## The oracle (dev-dependency only): oxideav-mp4

Per the container rule, `oxideav-mp4` is a **dev-dependency cross-validation oracle** and
never a production dependency. `tests/oracle.rs` opens the committed fixture with both our
`Mp4Demux`/`SampleTable` resolver and oxideav-mp4's `demux::open` and asserts the two
agree on sample count, per-sample size (== the mdat slice length), keyframe flag, and the
pts/dts arithmetic. A resolution bug (a wrong `stsc` fold, an off-by-one `stco`) shows up
as a disagreement against an independent reader.

## Progressive vs fragmented

- **Progressive** files (a single `moov` + `mdat`, `moov` before or after `mdat`) are the
  v1 target. The `Mp4Demux` constructor is handed the **file head through `moov`** (the
  same constructor-supplied-discovery model `MkvDemux` uses — a mid-pipeline element gets
  no input during preroll and the scheduler freezes topology after preroll), so the sample
  tables are resolved up front regardless of where `mdat` sits.
- **Fragmented** files (`moov` carrying `mvex`, then `moof`+`mdat` fragments; §8.8) are
  **detected and errored loudly** — the sample tables live in the fragments, not `stbl`,
  which is a materially different resolution path (out of v1 scope, tracked in `lib.rs`).

## Edit lists (`elst`, §8.6.6)

Only the common **single-entry media-time shift** is honoured: one `elst` entry with a
non-negative `media_time` shifts the whole track's presentation timeline by that offset
(subtracted from each sample's composition time so presentation starts at zero — the
"trim the encoder's priming/negative-cts" idiom). Multi-entry edit lists (dwell/empty
edits, rate changes, seek playlists) are **documented unsupported**: the raw sample
timeline is used unchanged and a bus `Warning` is posted. An empty edit (`media_time ==
-1`) as the sole entry is likewise ignored (its effect — an initial gap — needs an output
timeline the byte-bridge model does not carry yet).

## Not yet (v1 scope)

- **Fragmented MP4** (`moof`) — detected + errored; the resolver is `stbl`-only.
- **Multi-entry edit lists** — single media-time shift only (above).
- **Codec-init via caps** — like `MkvDemux`, the head bytes are a constructor argument
  while codec-init negotiation is built; the sample entry's `avcC`/`hvcC` is emitted as an
  Annex B parameter-set head before the first frame instead.
- **A muxer** — a separate follow-up (the box grammar is symmetric; a `writer.rs` joins
  later without reshaping the read side).
- **AAC (`mp4a`) decode** — the raw AAC access units are forwarded on a `bytes` pad; the
  `esds` `AudioSpecificConfig` is not parsed (no AAC decoder in-tree yet). The `aac`
  family name is reserved for when a decoder lands.
