# EBML + Matroska (MKV) — spec summary for `pf-mkv`

Interpretation notes and the normative element layout for the hand-written Matroska
**muxer**, per the plugin conventions (spec: First-party codecs — "the RFC/spec gets
checked into the crate's `spec/` dir" and "spec errata and interpretation decisions get
documented per crate"). Code cross-references this file (e.g. `§ID-tree`) and the two
normative sources below.

## Normative sources

- **RFC 8794** — "Extensible Binary Meta Language" (EBML), Oct 2020,
  `https://www.rfc-editor.org/rfc/rfc8794.txt`. Defines the byte grammar: Variable-Size
  Integers (VINTs), Element IDs, Element Data Sizes, master/child elements, the unknown
  size form. Sections cited as `RFC 8794 §4`, `§5`, `§6`.
- **Matroska** — the element ID tree and semantics, `https://www.matroska.org/`
  (`technical/elements.html`, `technical/basics.html`) and the codec mapping
  `technical/codec_specs.html`. The `A_FLAC` mapping defers framing to
  **RFC 9639 §10.2** ("FLAC-in-Ogg / native FLAC framing").

Matroska is *not* a byte-for-byte IETF wire spec with worked examples; it is an EBML
*schema* (a tree of typed elements keyed by ID). This file documents the exact subset the
muxer emits, at the byte level, so review means reading `ebml.rs` / `writer.rs` against
the grammar here.

---

## EBML grammar (RFC 8794)

Every element on the wire is `Element ID` · `Element Data Size` · `Element Data`
(RFC 8794 §5, §6). Both the ID and the size are **VINTs**; the data is either child
elements (a *master* element) or a typed scalar/binary (a *leaf*).

### VINT — Variable-Size Integer (RFC 8794 §4)

A VINT is 1–8 octets. It begins with `VINT_WIDTH` — zero or more `0` bits — terminated by
the `VINT_MARKER`, a single `1` bit (§4.1–4.2). The marker's position gives the total
octet length: `1xxxxxxx` = 1 octet, `01xxxxxx xxxxxxxx` = 2 octets, `001…` = 3 octets, and
so on up to 8. The remaining bits after the marker are `VINT_DATA`, big-endian, left
zero-padded (§4.3).

    length  leading bits   VINT_DATA bits   unsigned value range
    1       1              7                0 .. 2^7-1  (with the -1 rule, 0..2^7-2)
    2       01             14               0 .. 2^14-1
    3       001            21               0 .. 2^21-1
    ...
    8       00000001       56               0 .. 2^56-1

**Element Data Size** (RFC 8794 §6.1) is a VINT whose `VINT_DATA` is the length of the
element's data in octets. Unlike IDs it "is not mandated to be encoded at the shortest
valid length" (§6.1) — so the muxer is free to pick a width, e.g. a **fixed 8-octet size
reserved up front and back-patched** once a master element's length is known (see
"Sizing" below). One subtlety (§6.3): a value exactly equal to `2^(7n)-1` is the
all-ones "unknown" pattern at width `n`, so it MUST use `n+1` octets. This muxer sidesteps
it by using a fixed 8-octet size for the back-patched masters and shortest-length sizes
only for small leaves whose lengths are known and nowhere near that boundary.

**Unknown data size** (RFC 8794 §6.2): "An Element Data Size with all VINT_DATA bits set
to one is reserved as an indicator that the size of the Element is unknown." Only *master*
elements may use it, and only where the schema permits (Segment, Cluster). At width 1 that
byte is `0xFF`; the muxer emits the 1-octet form `0xFF` for streamed masters.

**Element ID** (RFC 8794 §5) is *also* VINT-encoded, but with extra constraints: the
`VINT_DATA` "MUST NOT be all `0` or all `1`" and "MUST be encoded at the shortest valid
length" (§5). Crucially, a Matroska ID is written **including** its VINT length-descriptor
bits — the ID *is* the whole VINT, marker and all (that is why e.g. the EBML Header ID is
the four bytes `1A 45 DF A3`, whose top byte `0x1A = 0b0001_1010` has the 4-octet marker
`0001`). So the muxer stores each ID as its **pre-encoded** big-endian byte string and
writes it verbatim; it does not re-derive the width. `ebml.rs` keeps the canonical IDs as
byte-array constants and `write_id` just appends them.

---

## Element ID tree — the subset this muxer emits  {#ID-tree}

IDs shown as their canonical on-the-wire hex (the full VINT, per above). `[m]` = master
(children follow), otherwise a typed leaf. Indentation = nesting. Values in parentheses are
what the muxer writes.

    1A45DFA3  EBML                         [m]  (the EBML Header)
      4286    EBMLVersion                  uint (1)
      42F7    EBMLReadVersion              uint (1)
      42F2    EBMLMaxIDLength              uint (4)
      42F3    EBMLMaxSizeLength            uint (8)
      4282    DocType                      str  ("matroska")
      4287    DocTypeVersion               uint (4)
      4285    DocTypeReadVersion           uint (2)

    18538067  Segment                      [m]  (unknown-size, streamed — see Sizing)
      1549A966  Info                       [m]
        2AD7B1  TimestampScale             uint (1_000_000 ns → ms tick)
        4D80    MuxingApp                  str  ("pf-mkv")
        5741    WritingApp                 str  ("pf-mkv")
      1654AE6B  Tracks                     [m]
        AE      TrackEntry                 [m]   (one per configured track)
          D7    TrackNumber                uint (1-based)
          73C5  TrackUID                   uint (== TrackNumber; a stable nonzero id)
          83    TrackType                  uint (1=video, 2=audio)
          86    CodecID                    str  ("A_FLAC" / "V_VP8" / …)
          63A2  CodecPrivate               bin  (fLaC+STREAMINFO, or avcC/hvcC; see below)
          9C    FlagLacing                 uint (0 — muxer emits one frame per block)
          E1    Audio                      [m]  (audio tracks)
            B5  SamplingFrequency          f32/f64 (Hz)
            9F  Channels                   uint
            6264 BitDepth                  uint (bits/sample)
          E0    Video                      [m]  (video tracks — instead of Audio)
            B0  PixelWidth                 uint (encoded frame width)
            BA  PixelHeight                uint (encoded frame height)
      1F43B675  Cluster                    [m]  (unknown-size, streamed — one per window)
        E7      Timestamp                  uint (cluster base, in TimestampScale ticks)
        A3      SimpleBlock                bin  (track VINT + s16 rel-ts + flags + frame)

IDs are grouped in `ebml.rs::id` exactly in this order.

### SimpleBlock body (Matroska `basics`/`block_structure`)  {#simpleblock}

A `SimpleBlock` (ID `0xA3`) is a binary element whose *data* is a self-describing block —
Matroska's lighter alternative to `BlockGroup`+`Block`, legal because FLAC frames are all
independently decodable (every FLAC frame is a keyframe). Layout of the data:

    Track Number    VINT             the block's track (VINT-encoded, 1-based)
    Timestamp       int16, signed    big-endian, relative to the Cluster Timestamp,
                                     in TimestampScale ticks
    Flags           u8               bit 0x80 = keyframe; lacing bits 0x06 = 00 (none)
    Frame data      bytes            the codec frame(s), here exactly one FLAC frame

The relative timestamp is **signed 16-bit**, so a block's timestamp must be within
±32767 ticks of its cluster's base — the muxer starts a new Cluster before that window
would overflow (and, for a track-0 keyframe, opportunistically). With the default
`TimestampScale = 1_000_000` (1 ms/tick) that window is ±32.767 s.

---

## `A_FLAC` codec mapping (Matroska `codec_specs`; RFC 9639 §10.2)

- **CodecID** = the string `"A_FLAC"`.
- **CodecPrivate** = "All FLAC data before the first audio frame"
  (`codec_specs.html`), i.e. the native FLAC stream head: the `fLaC` marker (4 bytes)
  followed by the metadata blocks — at minimum the STREAMINFO block *with its 4-byte
  metadata-block header* (§8.1). The muxer takes this blob verbatim from the caller (it is
  exactly the header bytes `FlacEncoder::new` emits, back-patched by `finish()`), so the
  bytes stored are bit-identical to a native `.flac` file's head.
- **Block payload** = native FLAC **frames**, one per `SimpleBlock` (RFC 9639 §10.2 native
  framing). No transformation: the encoder's frame bytes are copied straight into the block
  after the block header. Every FLAC frame is independently decodable, so every block is a
  keyframe (flag `0x80`).

The muxer is codec-agnostic beyond this table: `CodecID`, `CodecPrivate`, and the frame
bytes are all supplied by the caller/element, so the same writer muxes any
frame-per-block codec by changing the strings.

---

## Video codec mappings (RFC 9559 §12; Matroska codec registry)  {#video-codecs}

A video track sets `TrackType = 1` and carries a `Video` master (PixelWidth/PixelHeight)
instead of `Audio`. Two families, distinguished by how a Block's payload relates to the
codec's elementary stream:

- **WebM raw** — `V_VP8`, `V_VP9`, `V_AV1`. One Block is exactly one codec frame / temporal
  unit; there is **no CodecPrivate** (VP8/VP9 store no out-of-band config; AV1 optionally
  does but WebM commonly omits it). The demuxer forwards Block bytes **verbatim** and just
  names the pad family (`vp8`/`vp9`/`av1`) so a decoder links. pts is the Block timestamp.

- **ISO-BMFF NAL** — `V_MPEG4/ISO/AVC` (H.264) and `V_MPEGH/ISO/HEVC` (H.265). A Block holds
  **length-prefixed NAL units** (each NAL preceded by a big-endian length whose size, 1–4
  octets, is `lengthSizeMinusOne + 1` from the config record), and the parameter sets live in
  the CodecPrivate as an `AVCDecoderConfigurationRecord` / `HEVCDecoderConfigurationRecord`
  (ISO/IEC 14496-15). Our H.264/H.265 decoders consume **Annex B** (`00 00 00 01` start-code)
  streams, so the demuxer reframes (`codec.rs`): the parameter sets (SPS/PPS, VPS/SPS/PPS)
  become an Annex B head emitted once before the first frame, and each Block's length-prefixed
  NALs become start-code NALs, one access unit per output buffer. A malformed record/block
  warns-and-drops (untrusted input; never panics).

The `CodecID → announce family` map (`codec::family_for`) matches each decoder's sink offer:
`V_VP8→vp8`, `V_VP9→vp9`, `V_AV1→av1`, `V_MPEG4/ISO/AVC→h264/annexb`,
`V_MPEGH/ISO/HEVC→h265/annexb`, `A_FLAC→flac`, else `bytes`. The demux src pad also announces
`width`/`height` from the Video element (the decoder re-announces authoritative dims anyway).

The writer emits a video track with `TrackConfig::video`/`TrackConfig::vp8`; today's single-
sink-pad `MkvMux` element is FLAC-oriented, so a dedicated V_VP8 mux **element** is a
naming-only follow-up over the same N-track writer (see `lib.rs`, "Not yet").

---

## Sizing — how master-element lengths are handled  {#sizing}

Two length strategies, chosen per element (RFC 8794 §6):

1. **Back-patched, fixed 8-octet size** — used for the finite masters whose full contents
   the muxer buffers before flushing: the EBML Header, Info, Tracks (and each TrackEntry /
   Audio child within Tracks). The muxer writes the ID, reserves an 8-octet size slot
   (`0x01` + seven zero bytes, the canonical 8-octet-width VINT prefix), appends the
   children into the same output buffer, then patches the 56-bit `VINT_DATA` in place with
   the now-known byte length. Eight octets is always enough (max size `2^56-2`) and, being a
   fixed width, sidesteps the `2^(7n)-1` shortest-length edge case (§6.3). These elements
   are small and fully known, so buffering them is cheap.

2. **Unknown size (streamed)** — used for the two *open-ended* masters, `Segment` and
   `Cluster` (RFC 8794 §6.2). Their length is not known when their header is written (more
   Clusters / more blocks may follow), and a live muxer must not seek. The muxer writes the
   ID followed by the 1-octet unknown-size marker `0xFF`; the elements are terminated
   *implicitly* — a Cluster ends at the next Cluster's ID (or end of stream) and the Segment
   ends at end of stream. A parser knows a child ends the master when it reads an ID that
   the schema says is not a legal child of the open master (Matroska `basics`: "An element
   with unknown size is terminated by … an element that is not a valid child"). This is the
   standard streaming form (what `ffmpeg`/`mkvclean` emit for live/piped output) and needs
   no back-patching, so `finalize()` is a no-op flush.

So: the header block (EBML Header + Segment-open + Info + Tracks + Cluster-open) is emitted
once; every `write_frame` appends a `SimpleBlock` to the current Cluster or opens a new one;
`finalize()` has nothing to seek. The chosen design keeps the muxer **single-pass and
seek-free**, which is what a streaming media framework needs.
