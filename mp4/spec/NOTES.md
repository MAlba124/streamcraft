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
  - §8.10.1 User Data Box (`udta`) / §8.11.1 Meta Box (`meta`) — the metadata path walked by
    `src/ilst.rs`. **`meta` is a FullBox**: four octets of version+flags precede its children.
  - §12.1.3 VisualSampleEntry, §12.2.3 AudioSampleEntry — the `stsd` entry layouts.

- **Apple, *QuickTime File Format Specification*, "Metadata" chapter** (the published
  developer documentation; freely available, not vendored because it is Apple's text). The
  iTunes metadata *item* scheme `src/ilst.rs` reads is Apple's, not ISO's, and has no clause
  numbers — citations in that file read "(QTFF Metadata)". Used for: the metadata item list
  (`ilst`) and the item-id-as-box-type convention; the item data atom (`data`) with its
  4-octet type indicator (type-set octet + 3-octet well-known type) and 4-octet locale
  indicator; the well-known types 0 (implicit), 1 (UTF-8), 13 (JPEG), 14 (PNG), 21 (BE signed
  integer); the `©`-prefixed (0xA9) legacy user-data ids `©nam`/`©ART`/`©alb`/`©day`/`©gen`/
  `©wrt`/`©cmt` plus `aART`, `trkn`, `disk`, `covr`, `gnre`; and the `----` freeform triple
  `mean` / `name` / `data` that carries ReplayGain. Also used, in `props_from_moov`, for the
  QuickTime **version-2 sound description** layout (an `f64` sample rate at octet 32 and a u32
  channel count at octet 40), which is how a `.mov` states a rate the ISO 16.16 field cannot
  hold. Clean-room: written from these standards only — no tag-library source was consulted.

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

### The scanner's use of `elst` (`ilst::props_from_moov`)

Separate from the demuxer's shift above, the no-decode props path uses the edit list to answer
"how long does this file play". An MP4 states that three times — the sound track's `mdhd`
(media duration, fine timescale), its `elst` (presentation duration, coarse *movie* timescale),
and `mvhd` (the longest track's presentation duration) — and they disagree. Two guarded rules,
both measured against `ffprobe` over the 15 MP4/M4A files in the reference library:

* **The edit list wins over `mdhd` only when it differs by more than one movie tick.** A real
  priming trim (`media_time > 0`) moves the duration by tens of milliseconds. But the common
  `elst` is a single `media_time == 0` entry that trims nothing and merely restates `mdhd`
  rounded into the movie timescale — four of the 15 files are exactly that, and honouring them
  unconditionally puts the duration 0.17–0.87 ms *off* where `mdhd` alone is exact.
* **`mvhd` wins over the sound track only in a multi-track movie**, and again only past one
  movie tick. §8.2.2 defines `mvhd.duration` as the longest track's, so in a single-track file
  it *is* the sound track's and any excess is a stale header; in a movie with video it is the
  file's real playing time. This is what makes `ffmpeg_bar_test/output.mp4` report 427.734 s
  (`ffprobe`'s format duration is 427.733333) instead of the sound track's raw 427.745 s.

Empty edits (`media_time == -1`) contribute no duration in either rule: §8.6.6 makes them gaps,
and they present no media. Entry widths are handled at both versions (v0 32-bit, v1 64-bit).
The walk is hand-rolled rather than built on `boxes::parse_elst`, which returns a `Vec` — the
scanner path allocates nothing.

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
