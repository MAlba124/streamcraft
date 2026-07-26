//! `MatroskaWriter` — the muxer: track configs + `(track, timestamp, frame)` → a
//! well-formed Matroska byte stream (spec: `spec/MATROSKA.md`; RFC 8794 EBML). Written
//! against the element ID tree and sizing rules in the spec doc; code cross-references it
//! (`§ID-tree`, `§sizing`, `§simpleblock`).
//!
//! ## Multi-track by construction
//! The writer takes a `Vec` of [`TrackConfig`] and emits one `TrackEntry` per track. It is
//! not limited to one track — [`write_frame`](MatroskaWriter::write_frame) selects the
//! track by number. (The [`MkvMux`](crate::MkvMux) *element* exposes a single sink pad for
//! now; fanning several sink pads into a multi-track writer is a core-side follow-up, so
//! the writer is already N-track ready.)
//!
//! ## Sizing (spec `§sizing`)
//! Two length strategies, single-pass and seek-free:
//! - **Back-patched fixed 8-octet size** for the finite masters the writer buffers whole
//!   (EBML Header, Info, Tracks, each TrackEntry/Audio) — reserve an 8-octet slot, append
//!   children, patch the length in place. All in one output `Vec`, so no seeking.
//! - **Unknown size** (`0xFF`, RFC 8794 §6.2) for the open-ended `Segment` and each
//!   `Cluster`, terminated implicitly by the next Cluster / end of stream. So
//!   [`finalize`](MatroskaWriter::finalize) has nothing to patch — it is a flush only.
//!
//! ## SimpleBlock (spec `§simpleblock`)
//! Each frame is one `SimpleBlock`: track number (VINT), signed 16-bit timestamp relative
//! to the Cluster's base (in TimestampScale ticks), a flags byte (keyframe bit `0x80`, no
//! lacing), then the frame bytes copied verbatim. The relative timestamp is 16-bit signed,
//! so a new Cluster is started before a frame would fall outside `±32767` ticks of the
//! current base (and, opportunistically, on a track-0 keyframe).

use std::collections::VecDeque;

use streamcraft_core::memory::Memory;

use crate::ebml::{self, id};

/// The default Matroska TimestampScale (spec `§ID-tree`; Matroska `basics`): nanoseconds
/// per timestamp tick. `1_000_000` ns = 1 ms per tick — the Matroska default, giving a
/// `±32.767 s` SimpleBlock window on the signed-16-bit relative timestamp.
pub const DEFAULT_TIMESTAMP_SCALE: u64 = 1_000_000;

/// The muxer's identity, written into `MuxingApp`/`WritingApp` (spec `§ID-tree`).
pub const APP_NAME: &str = "sc-mkv";

/// TrackType for video (spec `§ID-tree`; RFC 9559 §5.1.4.1.3 Table 2: 1 video, 2 audio).
const TRACK_TYPE_VIDEO: u64 = 1;
/// TrackType for audio (spec `§ID-tree`; Matroska `basics`: 1 video, 2 audio).
const TRACK_TYPE_AUDIO: u64 = 2;

/// SimpleBlock flag: this block is a keyframe (spec `§simpleblock`). Bit `0x80`.
const BLOCK_FLAG_KEYFRAME: u8 = 0x80;

/// The signed-16-bit relative-timestamp span of one Cluster (spec `§simpleblock`). A frame
/// whose tick offset from the current Cluster base falls outside `[MIN, MAX]` forces a new
/// Cluster. We only ever produce non-negative offsets (base = first frame's tick), so the
/// practical limit is `MAX`.
const REL_TS_MAX: i64 = i16::MAX as i64;
const REL_TS_MIN: i64 = i16::MIN as i64;

/// Audio parameters for a track's `Audio` element (spec `§ID-tree`).
#[derive(Clone, Debug, PartialEq)]
pub struct AudioConfig {
    /// `SamplingFrequency` (Hz). Written as a 64-bit float (RFC 8794 §7.3).
    pub sampling_frequency: f64,
    /// `Channels`.
    pub channels: u32,
    /// `BitDepth` (bits per sample). Written when nonzero; `0` omits it (unknown).
    pub bit_depth: u32,
}

/// Video parameters for a track's `Video` element (RFC 9559 §5.1.4.1.28). Present only on a
/// video track; its presence is what makes [`MatroskaWriter`] emit a `Video` master and set
/// TrackType = video (1) instead of audio (2).
#[derive(Clone, Debug, PartialEq)]
pub struct VideoConfig {
    /// `PixelWidth` (RFC 9559 §5.1.4.1.28.6) — encoded frame width in pixels. MUST be nonzero.
    pub pixel_width: u32,
    /// `PixelHeight` (RFC 9559 §5.1.4.1.28.7) — encoded frame height in pixels. MUST be nonzero.
    pub pixel_height: u32,
    /// Optional `Video\Colour` colorimetry (RFC 9559 §5.1.4.1.31) — preserved on
    /// remux when the upstream announcement carried it.
    pub colour: Option<ColourConfig>,
}

/// The `Colour` element's children as ITU-T H.273 code points (RFC 9559
/// §5.1.4.1.31). A child equal to its RFC 9559 default — `2` ("unspecified")
/// for the H.273 points, `0` for `Range` — is not written. `0` cannot mark an
/// absent matrix: it is a *valid* code point (identity/RGB).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColourConfig {
    /// H.273 §8.3 `MatrixCoefficients`.
    pub matrix: u8,
    /// mkv `Range` (1 limited, 2 full).
    pub range: u8,
    /// H.273 §8.2 `TransferCharacteristics`.
    pub transfer: u8,
    /// H.273 §8.1 `Primaries`.
    pub primaries: u8,
}

impl Default for ColourConfig {
    /// All children at their RFC 9559 defaults — an empty config, nothing written.
    fn default() -> Self {
        Self { matrix: 2, range: 0, transfer: 2, primaries: 2 }
    }
}

impl ColourConfig {
    /// Whether anything at all would be written.
    pub fn is_empty(self) -> bool {
        self == Self::default()
    }
}

/// One track to mux (spec `§ID-tree` TrackEntry). The writer holds a `Vec` of these. A track
/// is audio unless it carries a [`VideoConfig`] (`video: Some`), in which case it is a video
/// track and the `audio` params are ignored.
#[derive(Clone, Debug)]
pub struct TrackConfig {
    /// `TrackNumber` (1-based, must be nonzero and unique). Also used as `TrackUID`.
    pub track_number: u64,
    /// `CodecID`, e.g. `"A_FLAC"` (audio) or `"V_VP8"` (video; RFC 9559 §12 codec mappings).
    pub codec_id: String,
    /// `CodecPrivate` — codec init data, e.g. FLAC `fLaC` + STREAMINFO, or an
    /// AVCDecoderConfigurationRecord for `V_MPEG4/ISO/AVC`. Empty means the element is omitted
    /// (WebM video — VP8/VP9/AV1 — stores frames raw with no CodecPrivate).
    pub codec_private: Vec<u8>,
    /// Audio parameters. Ignored (and typically zero) when `video` is `Some`.
    pub audio: AudioConfig,
    /// Video parameters when this is a video track, else `None` (an audio track).
    pub video: Option<VideoConfig>,
}

impl TrackConfig {
    /// An `A_FLAC` audio track from its number, native FLAC head (`fLaC` + STREAMINFO), and
    /// audio params. Convenience for the common case (spec: A_FLAC mapping).
    pub fn flac(track_number: u64, codec_private: Vec<u8>, sampling_frequency: f64, channels: u32, bit_depth: u32) -> Self {
        Self {
            track_number,
            codec_id: "A_FLAC".to_string(),
            codec_private,
            audio: AudioConfig { sampling_frequency, channels, bit_depth },
            video: None,
        }
    }

    /// A generic audio track from its number, `CodecID`, `CodecPrivate`, and audio params
    /// (RFC 9559 §12 codec mappings). The A_AAC case: `CodecPrivate` is the raw
    /// **AudioSpecificConfig** (ISO/IEC 14496-3 §1.6.2.1) and each frame is one raw AAC
    /// access unit — no ADTS framing (RFC 9559 §12: Matroska stores AAC "as is", the
    /// stream parameters live in CodecPrivate). `bit_depth` 0 omits `BitDepth` (lossy
    /// codecs have no meaningful one).
    pub fn audio(track_number: u64, codec_id: &str, codec_private: Vec<u8>, sampling_frequency: f64, channels: u32, bit_depth: u32) -> Self {
        Self {
            track_number,
            codec_id: codec_id.to_string(),
            codec_private,
            audio: AudioConfig { sampling_frequency, channels, bit_depth },
            video: None,
        }
    }

    /// A video track from its number, `CodecID`, optional `CodecPrivate`, and pixel dimensions
    /// (RFC 9559 §5.1.4.1.28). WebM codecs (`V_VP8`/`V_VP9`/`V_AV1`) store frames raw with an
    /// empty CodecPrivate; `V_MPEG4/ISO/AVC` / `V_MPEGH/ISO/HEVC` carry their configuration
    /// record here. Sets TrackType = video (1) and writes a `Video` master, no `Audio`.
    pub fn video(track_number: u64, codec_id: &str, codec_private: Vec<u8>, pixel_width: u32, pixel_height: u32) -> Self {
        Self {
            track_number,
            codec_id: codec_id.to_string(),
            codec_private,
            audio: AudioConfig { sampling_frequency: 0.0, channels: 0, bit_depth: 0 },
            video: Some(VideoConfig { pixel_width, pixel_height, colour: None }),
        }
    }

    /// A `V_VP8` (WebM) video track — pre-encoded VP8 frames, no CodecPrivate, given only its
    /// pixel dimensions. Convenience for the common WebM case (spec: RFC 9559 §12 codec
    /// mappings).
    pub fn vp8(track_number: u64, pixel_width: u32, pixel_height: u32) -> Self {
        Self::video(track_number, "V_VP8", Vec::new(), pixel_width, pixel_height)
    }
}

/// Errors from the writer. The only failure modes are misuse — the byte emission itself
/// cannot fail.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum WriteError {
    /// A frame was submitted for a track number no [`TrackConfig`] declared.
    UnknownTrack(u64),
    /// A frame arrived before [`MatroskaWriter::write_header`].
    HeaderNotWritten,
    /// `write_header` was called with no tracks, or with a duplicate/zero track number.
    BadTracks,
}

/// State of the Cluster currently being built (spec `§simpleblock`). A Cluster groups
/// blocks sharing a timestamp base; a new one starts when the relative-timestamp window
/// would overflow or on a track-0 keyframe.
struct OpenCluster {
    /// The Cluster's base timestamp, in TimestampScale ticks — the value written in its
    /// `Timestamp` child, and the origin the SimpleBlock relative timestamps are measured
    /// from.
    base_tick: i64,
    /// The Cluster's staged bytes, held until the Cluster closes so its **real size** can
    /// be written. Unknown-size (`0xFF`) Clusters are RFC 8794 §6.2-legal and libav
    /// accepts them, but real players do not — mpv's own demuxer rejects every block
    /// inside one ("Corrupt file detected") — so the writer stages one Cluster (≈ one
    /// GOP) and emits it sized, like every mainstream muxer. The *Segment* stays
    /// unknown-size (universally accepted; nothing to back-patch).
    ///
    /// Contiguous mode ([`MatroskaWriter::write_frame`]): the whole body — Timestamp
    /// child + full SimpleBlocks, headers *and* frame bytes. Scatter mode
    /// ([`MatroskaWriter::write_frame_scatter`], ZERO-COPY.md Stage 2): only the
    /// Timestamp child + the ~10-octet SimpleBlock headers; the frame bytes ride
    /// `pieces` as refcounted [`Memory`] slices, so the staged Cluster costs refcounts,
    /// not a second copy of one GOP.
    buf: Vec<u8>,
    /// Scatter-mode frame payloads, each pinned to the `buf` offset it must follow when
    /// the body is emitted (its SimpleBlock header's end). Empty in contiguous mode.
    pieces: Vec<(usize, Memory)>,
    /// Whether a keyframe of the cue track landed in this Cluster — what makes it
    /// cue-worthy (a CuePoint must reference a position decoding can start from;
    /// mkvmerge cues video keyframes the same way).
    has_cue_key: bool,
}

impl OpenCluster {
    /// The Cluster's total body size — staged bytes plus every scatter payload. This is
    /// the **known size** written after the Cluster ID on close (the mpv-load-bearing
    /// sized-Cluster contract), available without making the body contiguous.
    fn body_size(&self) -> u64 {
        self.buf.len() as u64 + self.pieces.iter().map(|(_, m)| m.len() as u64).sum::<u64>()
    }
}

/// Multi-track Matroska muxer. Configure with tracks, [`write_header`], then
/// [`write_frame`] per encoded frame, then [`finalize`]. Everything appends into a
/// caller-owned `Vec<u8>` so steady-state muxing does not reallocate (spec: performance
/// first) — mirrors the `sc-ogg` writer's buffer-append contract.
pub struct MatroskaWriter {
    tracks: Vec<TrackConfig>,
    timestamp_scale: u64,
    /// Set once [`write_header`] has run — [`write_frame`] requires it.
    header_written: bool,
    /// The Cluster in progress, or `None` before the first frame / after a flush point.
    cluster: Option<OpenCluster>,
    /// The track number treated as the "keyframe anchor" for opening Clusters: the first
    /// configured track. A keyframe on it opens a fresh Cluster (typical Matroska cadence).
    anchor_track: u64,
    /// Presentation duration in ns, written as `Info\Duration` when known **before**
    /// [`write_header`] (a remux knows it from the source's tables; the header is written
    /// lazily at the first frame, so no back-patching is ever needed). Without it, players
    /// treat the unknown-size Segment as a live stream with no duration or seek bar.
    duration_ns: Option<u64>,
    /// Cue tracking (RFC 9559 §5.1.5), opt-in via [`enable_cues`](Self::enable_cues) —
    /// `None` keeps the writer strictly single-pass and seek-free (the default).
    cues: Option<CueState>,
    /// [`reserve_duration`](Self::reserve_duration) was called: the header writes a
    /// placeholder `Info\Duration` whose stream offset lands in `duration_at`.
    duration_reserved: bool,
    /// Stream offset of the reserved Duration *element* (ID byte), set by the header
    /// write. Same origin as [`SeekHeadPatch::offset`]: the first byte the writer ever
    /// produced (== the file offset when the stream is written from 0).
    duration_at: Option<u64>,
}

/// Cue/SeekHead bookkeeping (RFC 9559 §5.1.1, §5.1.5) — filled while writing, spent at
/// finalize. All "positions" are Segment Positions: byte offsets relative to the first
/// byte of the Segment's data (RFC 9559 §4), which is exactly where the SeekHead
/// reservation sits (it is the Segment's first child).
struct CueState {
    /// Absolute stream offset of the Segment data start == the SeekHead reservation.
    segment_data_start: u64,
    /// Segment Positions of the Info and Tracks masters (for the SeekHead).
    info_pos: u64,
    tracks_pos: u64,
    /// The track CuePoints reference: the first video track, else the anchor track.
    cue_track: u64,
    /// One `(CueTime ticks, CueClusterPosition)` per cue-worthy Cluster, in order.
    points: Vec<(u64, u64)>,
}

/// The fixed size of the SeekHead reservation (a Void the writer back-patches at
/// finalize): SeekHead master (4-octet ID + 8-octet size) + three Seek entries (Info,
/// Tracks, Cues), each ≤ 29 octets with 8-octet back-patched sizes — ≤ 99 total, so 128
/// leaves room and keeps the remainder big enough for the trailing Void header.
const SEEKHEAD_RESERVE: usize = 128;

/// The back-patch [`MatroskaWriter::finalize_scatter`] returns when cues are enabled:
/// `bytes` (exactly [`SEEKHEAD_RESERVE`] long — the SeekHead plus a Void filler)
/// replace the reservation at absolute stream offset `offset`. The stream itself stays
/// single-pass; this is the one positioned overwrite an indexed container needs.
pub struct SeekHeadPatch {
    pub offset: u64,
    pub bytes: Vec<u8>,
}

impl MatroskaWriter {
    /// A writer for `tracks` at the default TimestampScale (1 ms/tick). Track configs are
    /// validated in [`write_header`].
    pub fn new(tracks: Vec<TrackConfig>) -> Self {
        Self::with_timestamp_scale(tracks, DEFAULT_TIMESTAMP_SCALE)
    }

    /// A writer with an explicit TimestampScale in ns/tick (spec `§ID-tree`
    /// TimestampScale). Smaller ticks give finer timestamps but a narrower per-Cluster
    /// window (the SimpleBlock relative timestamp is signed 16-bit — spec `§simpleblock`).
    pub fn with_timestamp_scale(tracks: Vec<TrackConfig>, timestamp_scale: u64) -> Self {
        let anchor_track = tracks.first().map(|t| t.track_number).unwrap_or(1);
        Self {
            tracks,
            timestamp_scale,
            header_written: false,
            cluster: None,
            anchor_track,
            duration_ns: None,
            cues: None,
            duration_reserved: false,
            duration_at: None,
        }
    }

    /// Track cue points and emit a `Cues` master + front `SeekHead` (RFC 9559 §5.1.1,
    /// §5.1.5). Must be called before [`write_header`]. The header then reserves
    /// [`SEEKHEAD_RESERVE`] octets of Void as the Segment's first child; finalize emits
    /// the Cues after the last Cluster and produces the SeekHead bytes for that
    /// reservation — [`finalize`](Self::finalize) patches them in place (the whole
    /// stream is one `Vec`), [`finalize_scatter`](Self::finalize_scatter) returns a
    /// [`SeekHeadPatch`] for the caller's sink to apply as one positioned write.
    /// Without cues a player can only seek by scanning/bisecting Clusters.
    pub fn enable_cues(&mut self) {
        if self.header_written {
            return; // too late — the reservation must precede Info/Tracks
        }
        let cue_track = self
            .tracks
            .iter()
            .find(|t| t.video.is_some())
            .map(|t| t.track_number)
            .unwrap_or(self.anchor_track);
        self.cues = Some(CueState {
            segment_data_start: 0,
            info_pos: 0,
            tracks_pos: 0,
            cue_track,
            points: Vec::new(),
        });
    }

    /// Declare the presentation duration (ns), to be written as `Info\Duration`. Must be
    /// called before [`write_header`] (which the muxer element defers to the first frame);
    /// later calls are ignored — the Info master is already emitted.
    pub fn set_duration_ns(&mut self, ns: u64) {
        if !self.header_written {
            self.duration_ns = Some(ns);
        }
    }

    /// Reserve a back-patchable `Info\Duration` (RFC 9559 §5.1.2) for streams whose
    /// length is unknown until they end — a live recording writes its header at the
    /// first frame but only learns the duration at the cut. Must be called before
    /// [`write_header`]; the header then carries a placeholder (0.0) the caller
    /// replaces via [`duration_patch`](Self::duration_patch)'s positioned write. A
    /// known-upfront duration ([`set_duration_ns`](Self::set_duration_ns)) wins over
    /// the reservation.
    pub fn reserve_duration(&mut self) {
        if !self.header_written {
            self.duration_reserved = true;
        }
    }

    /// The positioned write replacing the reserved Duration placeholder with `ns`
    /// (in TimestampScale ticks, as a float — RFC 9559 §5.1.2). `None` without a
    /// reservation or before the header. Offsets share [`SeekHeadPatch`]'s origin
    /// (the stream's first byte), so a file written from offset 0 applies it as-is.
    pub fn duration_patch(&self, ns: u64) -> Option<SeekHeadPatch> {
        let offset = self.duration_at?;
        let mut bytes = Vec::with_capacity(16);
        ebml::write_f64(&mut bytes, id::DURATION, ns as f64 / self.timestamp_scale.max(1) as f64);
        Some(SeekHeadPatch { offset, bytes })
    }

    pub fn timestamp_scale(&self) -> u64 {
        self.timestamp_scale
    }

    pub fn tracks(&self) -> &[TrackConfig] {
        &self.tracks
    }

    fn validate_tracks(&self) -> Result<(), WriteError> {
        if self.tracks.is_empty() {
            return Err(WriteError::BadTracks);
        }
        for (i, t) in self.tracks.iter().enumerate() {
            if t.track_number == 0 {
                return Err(WriteError::BadTracks);
            }
            if self.tracks[..i].iter().any(|o| o.track_number == t.track_number) {
                return Err(WriteError::BadTracks); // duplicate track number
            }
        }
        Ok(())
    }

    /// Convert a nanosecond timestamp to TimestampScale ticks (spec `§ID-tree`). Rounds to
    /// nearest tick.
    fn ns_to_ticks(&self, timestamp_ns: u64) -> i64 {
        let scale = self.timestamp_scale.max(1);
        ((timestamp_ns + scale / 2) / scale) as i64
    }

    /// Write the stream head into `out` (spec `§ID-tree`): EBML Header, then the *open*
    /// Segment (unknown size), Info, and Tracks. After this, [`write_frame`] appends
    /// Clusters. Must be called exactly once, before any frame.
    pub fn write_header(&mut self, out: &mut Vec<u8>) -> Result<(), WriteError> {
        self.validate_tracks()?;
        self.write_ebml_header(out);
        // Segment: open with an unknown size — it runs to end of stream (spec `§sizing`).
        ebml::write_id(out, id::SEGMENT);
        ebml::write_unknown_size(out);
        if let Some(cues) = &mut self.cues {
            // The SeekHead reservation, as the Segment's first child (RFC 9559 §5.1.1
            // recommends the SeekHead be the first element): a Void (RFC 8794 §11.3.2)
            // finalize overwrites. Positions recorded here are `out`-relative; the
            // scatter path rebases them to absolute stream offsets after the call.
            cues.segment_data_start = out.len() as u64;
            ebml::write_id(out, id::VOID);
            ebml::write_size(out, (SEEKHEAD_RESERVE - 2) as u64); // 1-octet size VINT
            out.resize(out.len() + SEEKHEAD_RESERVE - 2, 0);
            cues.info_pos = out.len() as u64 - cues.segment_data_start;
        }
        self.write_info(out);
        if let Some(cues) = &mut self.cues {
            cues.tracks_pos = out.len() as u64 - cues.segment_data_start;
        }
        self.write_tracks(out);
        self.header_written = true;
        Ok(())
    }

    /// The EBML Header (spec `§ID-tree`; RFC 8794 §11.2.4), a fully-buffered master with a
    /// back-patched 8-octet size.
    fn write_ebml_header(&self, out: &mut Vec<u8>) {
        ebml::write_id(out, id::EBML);
        let size_at = ebml::reserve_size(out);
        let body_start = out.len();
        ebml::write_uint(out, id::EBML_VERSION, 1);
        ebml::write_uint(out, id::EBML_READ_VERSION, 1);
        ebml::write_uint(out, id::EBML_MAX_ID_LENGTH, 4);
        ebml::write_uint(out, id::EBML_MAX_SIZE_LENGTH, 8);
        ebml::write_string(out, id::DOC_TYPE, "matroska");
        ebml::write_uint(out, id::DOC_TYPE_VERSION, 4);
        ebml::write_uint(out, id::DOC_TYPE_READ_VERSION, 2);
        let len = (out.len() - body_start) as u64;
        ebml::patch_size(out, size_at, len);
    }

    /// The Segment `Info` master (spec `§ID-tree`), back-patched.
    fn write_info(&mut self, out: &mut Vec<u8>) {
        ebml::write_id(out, id::INFO);
        let size_at = ebml::reserve_size(out);
        let body_start = out.len();
        ebml::write_uint(out, id::TIMESTAMP_SCALE, self.timestamp_scale);
        // Duration is a float in TimestampScale ticks (RFC 9559 §5.1.2); without it a
        // player treats the unknown-size Segment as a live, duration-less stream.
        if let Some(ns) = self.duration_ns {
            ebml::write_f64(out, id::DURATION, ns as f64 / self.timestamp_scale.max(1) as f64);
        } else if self.duration_reserved {
            // A fixed-width (8-octet float) placeholder; `write_f64` is byte-stable, so
            // [`duration_patch`](Self::duration_patch) re-encodes the whole element at
            // this offset once the true duration is known. Recorded `out`-relative here;
            // the scatter path rebases to the absolute stream offset (like the cues).
            self.duration_at = Some(out.len() as u64);
            ebml::write_f64(out, id::DURATION, 0.0);
        }
        ebml::write_string(out, id::MUXING_APP, APP_NAME);
        ebml::write_string(out, id::WRITING_APP, APP_NAME);
        let len = (out.len() - body_start) as u64;
        ebml::patch_size(out, size_at, len);
    }

    /// The `Tracks` master with one `TrackEntry` per configured track (spec `§ID-tree`),
    /// back-patched.
    fn write_tracks(&self, out: &mut Vec<u8>) {
        ebml::write_id(out, id::TRACKS);
        let size_at = ebml::reserve_size(out);
        let body_start = out.len();
        for track in &self.tracks {
            Self::write_track_entry(out, track);
        }
        let len = (out.len() - body_start) as u64;
        ebml::patch_size(out, size_at, len);
    }

    /// One `TrackEntry` (spec `§ID-tree`), back-patched, with a nested `Audio` *or* `Video`
    /// master depending on whether the track carries a [`VideoConfig`].
    fn write_track_entry(out: &mut Vec<u8>, track: &TrackConfig) {
        ebml::write_id(out, id::TRACK_ENTRY);
        let size_at = ebml::reserve_size(out);
        let body_start = out.len();

        let track_type = if track.video.is_some() { TRACK_TYPE_VIDEO } else { TRACK_TYPE_AUDIO };
        ebml::write_uint(out, id::TRACK_NUMBER, track.track_number);
        ebml::write_uint(out, id::TRACK_UID, track.track_number); // stable nonzero UID
        ebml::write_uint(out, id::TRACK_TYPE, track_type);
        // Lacing off: the muxer emits exactly one frame per SimpleBlock (spec `§simpleblock`).
        ebml::write_uint(out, id::FLAG_LACING, 0);
        ebml::write_string(out, id::CODEC_ID, &track.codec_id);
        if !track.codec_private.is_empty() {
            ebml::write_binary(out, id::CODEC_PRIVATE, &track.codec_private);
        }

        match &track.video {
            Some(video) => Self::write_video(out, video),
            None => Self::write_audio(out, &track.audio),
        }

        let len = (out.len() - body_start) as u64;
        ebml::patch_size(out, size_at, len);
    }

    /// The nested `Audio` master (spec `§ID-tree`), itself back-patched.
    fn write_audio(out: &mut Vec<u8>, audio: &AudioConfig) {
        ebml::write_id(out, id::AUDIO);
        let audio_size_at = ebml::reserve_size(out);
        let audio_start = out.len();
        ebml::write_f64(out, id::SAMPLING_FREQUENCY, audio.sampling_frequency);
        ebml::write_uint(out, id::CHANNELS, audio.channels as u64);
        if audio.bit_depth != 0 {
            ebml::write_uint(out, id::BIT_DEPTH, audio.bit_depth as u64);
        }
        let audio_len = (out.len() - audio_start) as u64;
        ebml::patch_size(out, audio_size_at, audio_len);
    }

    /// The nested `Video` master (RFC 9559 §5.1.4.1.28), back-patched — PixelWidth and
    /// PixelHeight, the minimum a demuxer needs to seed a video track's format announcement.
    fn write_video(out: &mut Vec<u8>, video: &VideoConfig) {
        ebml::write_id(out, id::VIDEO);
        let video_size_at = ebml::reserve_size(out);
        let video_start = out.len();
        ebml::write_uint(out, id::PIXEL_WIDTH, video.pixel_width as u64);
        ebml::write_uint(out, id::PIXEL_HEIGHT, video.pixel_height as u64);
        // Optional Colour master (RFC 9559 §5.1.4.1.31): H.273 code points,
        // written only for children off their RFC 9559 defaults — remux
        // preserves color, an unspecified child stays implicit.
        if let Some(c) = video.colour.filter(|c| !c.is_empty()) {
            let dflt = ColourConfig::default();
            ebml::write_id(out, id::COLOUR);
            let colour_size_at = ebml::reserve_size(out);
            let colour_start = out.len();
            if c.matrix != dflt.matrix {
                ebml::write_uint(out, id::MATRIX_COEFFICIENTS, c.matrix as u64);
            }
            if c.range != dflt.range {
                ebml::write_uint(out, id::COLOUR_RANGE, c.range as u64);
            }
            if c.transfer != dflt.transfer {
                ebml::write_uint(out, id::TRANSFER_CHARACTERISTICS, c.transfer as u64);
            }
            if c.primaries != dflt.primaries {
                ebml::write_uint(out, id::PRIMARIES, c.primaries as u64);
            }
            let colour_len = (out.len() - colour_start) as u64;
            ebml::patch_size(out, colour_size_at, colour_len);
        }
        let video_len = (out.len() - video_start) as u64;
        ebml::patch_size(out, video_size_at, video_len);
    }

    /// Append one encoded frame as a `SimpleBlock` (spec `§simpleblock`), opening a new
    /// Cluster first if needed. `timestamp_ns` is the frame's presentation time in
    /// nanoseconds; it is converted to TimestampScale ticks. `keyframe` sets the block's
    /// keyframe flag.
    ///
    /// A new Cluster is opened when: there is none yet; the frame's tick offset from the
    /// current Cluster base would fall outside the signed-16-bit window; or the frame is a
    /// keyframe on the anchor (first) track. Because a Cluster is closed *implicitly* (by
    /// the next Cluster's ID), opening one needs no back-patch of the previous one.
    pub fn write_frame(
        &mut self,
        out: &mut Vec<u8>,
        track: u64,
        timestamp_ns: u64,
        bytes: &[u8],
        keyframe: bool,
    ) -> Result<(), WriteError> {
        let tick = self.frame_tick(track, timestamp_ns)?;
        if self.need_new_cluster(tick, track, keyframe) {
            self.open_cluster(out, tick);
        }

        // Safe: a Cluster is open now (just opened, or already was). Blocks stage into the
        // Cluster's buffer; `out` receives the whole sized Cluster when it closes.
        let cue_key = keyframe && self.cues.as_ref().is_some_and(|c| c.cue_track == track);
        let cluster = self.cluster.as_mut().expect("cluster open");
        if cue_key {
            cluster.has_cue_key = true;
        }
        let rel = (tick - cluster.base_tick) as i16; // within the checked window
        Self::write_simple_block(&mut cluster.buf, track, rel, bytes, keyframe);
        Ok(())
    }

    /// Scatter-mode [`write_frame`](Self::write_frame) (ZERO-COPY.md Stage 2): stages
    /// only the ~10-octet SimpleBlock header; the frame's bytes ride as the refcounted
    /// [`Memory`] itself, forwarded when the Cluster closes — no copy into the staging
    /// buffer, no copy out. Cluster boundaries, sizing and timestamps are byte-identical
    /// to [`write_frame`]: the mpv-load-bearing **known-size** Cluster is preserved,
    /// because the size only needs the running header+payload total
    /// ([`OpenCluster::body_size`]), not a contiguous body.
    pub fn write_frame_scatter(
        &mut self,
        out: &mut MuxOut,
        track: u64,
        timestamp_ns: u64,
        payload: Memory,
        keyframe: bool,
    ) -> Result<(), WriteError> {
        let tick = self.frame_tick(track, timestamp_ns)?;
        if self.need_new_cluster(tick, track, keyframe) {
            self.open_cluster_scatter(out, tick);
        }

        let cue_key = keyframe && self.cues.as_ref().is_some_and(|c| c.cue_track == track);
        let cluster = self.cluster.as_mut().expect("cluster open");
        if cue_key {
            cluster.has_cue_key = true;
        }
        let rel = (tick - cluster.base_tick) as i16; // within the checked window
        Self::write_simple_block_header(&mut cluster.buf, track, rel, payload.len(), keyframe);
        cluster.pieces.push((cluster.buf.len(), payload));
        Ok(())
    }

    /// [`write_header`](Self::write_header) into a scatter output: the head bytes become
    /// one queued byte run.
    pub fn write_header_scatter(&mut self, out: &mut MuxOut) -> Result<(), WriteError> {
        let stream_at = out.emitted();
        let mut local_start = 0u64;
        let r = out.append_bytes_with(|b| {
            local_start = b.len() as u64;
            self.write_header(b)
        });
        if r.is_ok() {
            if let Some(cues) = &mut self.cues {
                // `write_header` recorded the reservation's offset local to the byte
                // arena; rebase it to the absolute stream offset (the arena may hold
                // earlier runs). The Info/Tracks positions are Segment-relative and
                // need no rebasing.
                cues.segment_data_start = stream_at + (cues.segment_data_start - local_start);
            }
            if let Some(at) = &mut self.duration_at {
                // Same rebase for the Duration reservation (see `write_info`).
                *at = stream_at + (*at - local_start);
            }
        }
        r
    }

    /// Record a cue point for the Cluster being emitted at absolute stream offset
    /// `cluster_at` (RFC 9559 §5.1.5): its base time in ticks, its Segment Position.
    /// Only cue-worthy Clusters (a keyframe of the cue track landed in them) are
    /// recorded — a CuePoint must reference a position decoding can start from.
    fn record_cue(&mut self, c: &OpenCluster, cluster_at: u64) {
        if !c.has_cue_key {
            return;
        }
        if let Some(cues) = &mut self.cues {
            cues.points.push((c.base_tick.max(0) as u64, cluster_at - cues.segment_data_start));
        }
    }

    /// The per-frame preconditions shared by both write modes: header written, track
    /// known, timestamp quantized to TimestampScale ticks (round-to-nearest).
    fn frame_tick(&self, track: u64, timestamp_ns: u64) -> Result<i64, WriteError> {
        if !self.header_written {
            return Err(WriteError::HeaderNotWritten);
        }
        if !self.tracks.iter().any(|t| t.track_number == track) {
            return Err(WriteError::UnknownTrack(track));
        }
        Ok(self.ns_to_ticks(timestamp_ns))
    }

    /// Whether this frame opens a new Cluster (spec `§simpleblock`): none open yet, the
    /// tick offset would leave the signed-16-bit window, or a keyframe on the anchor
    /// (first) track.
    fn need_new_cluster(&self, tick: i64, track: u64, keyframe: bool) -> bool {
        match &self.cluster {
            None => true,
            Some(c) => {
                let rel = tick - c.base_tick;
                rel < REL_TS_MIN || rel > REL_TS_MAX
                    || (keyframe && track == self.anchor_track && rel != 0)
            }
        }
    }

    /// A fresh [`OpenCluster`] at `base_tick`, reusing the previous cluster's (cleared)
    /// allocations. Its body starts with the `Timestamp` child.
    fn opened(base_tick: i64, mut buf: Vec<u8>, pieces: Vec<(usize, Memory)>) -> OpenCluster {
        // Cluster base timestamp, in ticks (non-negative for a forward stream).
        ebml::write_uint(&mut buf, id::TIMESTAMP, base_tick.max(0) as u64);
        OpenCluster { base_tick, buf, pieces, has_cue_key: false }
    }

    /// Open a new Cluster at `base_tick` (spec `§ID-tree` Cluster, `§sizing`), first
    /// emitting the previous Cluster (if any) into `out` with its now-known size. The new
    /// Cluster's body stages in memory until it closes (see [`OpenCluster::buf`] —
    /// players reject unknown-size Clusters).
    fn open_cluster(&mut self, out: &mut Vec<u8>, base_tick: i64) {
        let (buf, pieces) = self.close_cluster(out);
        self.cluster = Some(Self::opened(base_tick, buf, pieces));
    }

    /// [`open_cluster`](Self::open_cluster) into a scatter output.
    fn open_cluster_scatter(&mut self, out: &mut MuxOut, base_tick: i64) {
        let (buf, pieces) = self.close_cluster_scatter(out);
        self.cluster = Some(Self::opened(base_tick, buf, pieces));
    }

    /// Emit the staged Cluster (if any) into `out` — ID, its **real** body size, then the
    /// body: staged byte runs interleaved with any scatter payloads (copied here, since a
    /// contiguous caller wants contiguous bytes). Returns the drained staging allocations
    /// for reuse (empty if no Cluster was open).
    fn close_cluster(&mut self, out: &mut Vec<u8>) -> (Vec<u8>, Vec<(usize, Memory)>) {
        let Some(mut c) = self.cluster.take() else { return (Vec::new(), Vec::new()) };
        // Contiguous mode: `out` is the whole stream, so its length *is* the Cluster's
        // absolute offset (the cue position, RFC 9559 §5.1.5.1.2).
        self.record_cue(&c, out.len() as u64);
        ebml::write_id(out, id::CLUSTER);
        ebml::write_size(out, c.body_size());
        let mut cur = 0;
        for (at, mem) in c.pieces.drain(..) {
            out.extend_from_slice(&c.buf[cur..at]);
            out.extend_from_slice(mem.data());
            cur = at;
        }
        out.extend_from_slice(&c.buf[cur..]);
        c.buf.clear();
        (c.buf, c.pieces)
    }

    /// Emit the staged Cluster (if any) into a scatter output (ZERO-COPY.md Stage 2):
    /// the Cluster head + staged header runs are appended as byte runs (tiny — ~10
    /// octets per block), while each payload [`Memory`] moves out as its own piece — a
    /// refcount move, the frame bytes are never copied. Returns the drained staging
    /// allocations for reuse.
    fn close_cluster_scatter(&mut self, out: &mut MuxOut) -> (Vec<u8>, Vec<(usize, Memory)>) {
        let Some(mut c) = self.cluster.take() else { return (Vec::new(), Vec::new()) };
        self.record_cue(&c, out.emitted());
        let body_size = c.body_size();
        out.append_bytes_with(|b| {
            ebml::write_id(b, id::CLUSTER);
            ebml::write_size(b, body_size);
        });
        let mut cur = 0;
        for (at, mem) in c.pieces.drain(..) {
            if at > cur {
                out.append_bytes_with(|b| b.extend_from_slice(&c.buf[cur..at]));
                cur = at;
            }
            out.push_payload(mem);
        }
        if cur < c.buf.len() {
            out.append_bytes_with(|b| b.extend_from_slice(&c.buf[cur..]));
        }
        c.buf.clear();
        (c.buf, c.pieces)
    }

    /// Serialise one `SimpleBlock` element (spec `§simpleblock`): the `A3` ID, the data
    /// size (the whole block body), then the body — track-number VINT, big-endian signed
    /// 16-bit relative timestamp, flags byte, and the frame bytes verbatim.
    fn write_simple_block(out: &mut Vec<u8>, track: u64, rel_ts: i16, frame: &[u8], keyframe: bool) {
        Self::write_simple_block_header(out, track, rel_ts, frame.len(), keyframe);
        // Frame payload, verbatim (zero-transform — spec: A_FLAC frames stored natively).
        out.extend_from_slice(frame);
    }

    /// Everything of a `SimpleBlock` **except** the frame bytes (spec `§simpleblock`):
    /// ID, body size (accounting the frame's length), track-number VINT, big-endian
    /// signed 16-bit relative timestamp, flags byte — ~10 octets. What scatter mode
    /// stages per block (ZERO-COPY.md Stage 2).
    fn write_simple_block_header(
        out: &mut Vec<u8>,
        track: u64,
        rel_ts: i16,
        frame_len: usize,
        keyframe: bool,
    ) {
        ebml::write_id(out, id::SIMPLE_BLOCK);
        // Body length: track VINT (its size-form encodes the number) + 2 (ts) + 1 (flags)
        // + frame. The track number is written with the *size* VINT form (a plain VINT of
        // the value), per the block grammar.
        let track_vint_len = ebml::vint_size_len(track);
        let body_len = track_vint_len + 2 + 1 + frame_len;
        ebml::write_size(out, body_len as u64);

        // Track number as a VINT (spec `§simpleblock`).
        ebml::write_size(out, track);
        // Relative timestamp: signed 16-bit, big-endian.
        out.extend_from_slice(&rel_ts.to_be_bytes());
        // Flags: keyframe bit; lacing bits 0 (no lacing).
        out.push(if keyframe { BLOCK_FLAG_KEYFRAME } else { 0 });
    }

    /// Finish the stream (spec `§sizing`): emit the final staged Cluster with its real
    /// size. The `Segment` stays an unknown-size master closed implicitly at end of
    /// stream — nothing already emitted is back-patched, so the writer remains
    /// single-pass and seek-free — **unless** cues are enabled: then the `Cues` master
    /// is appended after the last Cluster and the SeekHead reservation is patched *in
    /// place* (contiguous mode owns the whole stream in `out`, so the patch is a
    /// memcpy, not a seek). `out` must be the same `Vec` the header was written into.
    pub fn finalize(&mut self, out: &mut Vec<u8>) {
        let _ = self.close_cluster(out);
        if let Some(cues) = self.cues.take() {
            if cues.points.is_empty() {
                return; // nothing cue-worthy — the reservation stays a valid Void
            }
            let cues_pos = out.len() as u64 - cues.segment_data_start;
            write_cues(out, cues.cue_track, &cues.points);
            let sh = build_seekhead(&cues, cues_pos);
            let at = cues.segment_data_start as usize;
            out[at..at + sh.len()].copy_from_slice(&sh);
        }
    }

    /// [`finalize`](Self::finalize) into a scatter output. With cues enabled, appends
    /// the `Cues` master and returns the [`SeekHeadPatch`] for the front reservation —
    /// the caller's byte sink applies it as one positioned write (the scatter stream
    /// has already left the writer, so it cannot patch in place).
    pub fn finalize_scatter(&mut self, out: &mut MuxOut) -> Option<SeekHeadPatch> {
        let _ = self.close_cluster_scatter(out);
        let cues = self.cues.take()?;
        if cues.points.is_empty() {
            return None; // nothing cue-worthy — the reservation stays a valid Void
        }
        let cues_pos = out.emitted() - cues.segment_data_start;
        out.append_bytes_with(|b| write_cues(b, cues.cue_track, &cues.points));
        Some(SeekHeadPatch {
            offset: cues.segment_data_start,
            bytes: build_seekhead(&cues, cues_pos),
        })
    }
}

/// The `Cues` master (RFC 9559 §5.1.5): one CuePoint per cue-worthy Cluster —
/// `CueTime` (ticks) and `CueTrackPositions { CueTrack, CueClusterPosition }`, the
/// position being a Segment Position (RFC 9559 §4). Emitted after the last Cluster;
/// discovered via the front SeekHead.
fn write_cues(out: &mut Vec<u8>, cue_track: u64, points: &[(u64, u64)]) {
    ebml::write_id(out, id::CUES);
    let size_at = ebml::reserve_size(out);
    let body = out.len();
    for &(time, pos) in points {
        ebml::write_id(out, id::CUE_POINT);
        let p_at = ebml::reserve_size(out);
        let p0 = out.len();
        ebml::write_uint(out, id::CUE_TIME, time);
        ebml::write_id(out, id::CUE_TRACK_POSITIONS);
        let tp_at = ebml::reserve_size(out);
        let tp0 = out.len();
        ebml::write_uint(out, id::CUE_TRACK, cue_track);
        ebml::write_uint(out, id::CUE_CLUSTER_POSITION, pos);
        let tp_len = (out.len() - tp0) as u64;
        ebml::patch_size(out, tp_at, tp_len);
        let p_len = (out.len() - p0) as u64;
        ebml::patch_size(out, p_at, p_len);
    }
    let len = (out.len() - body) as u64;
    ebml::patch_size(out, size_at, len);
}

/// The bytes that replace the [`SEEKHEAD_RESERVE`]-octet reservation exactly: a
/// SeekHead pointing at Info, Tracks and the just-emitted Cues (RFC 9559 §5.1.1),
/// then a Void filling the remainder (RFC 8794 §11.3.2).
fn build_seekhead(cues: &CueState, cues_pos: u64) -> Vec<u8> {
    let mut sh = Vec::with_capacity(SEEKHEAD_RESERVE);
    ebml::write_id(&mut sh, id::SEEK_HEAD);
    let size_at = ebml::reserve_size(&mut sh);
    let body = sh.len();
    let targets: [(&[u8], u64); 3] =
        [(id::INFO, cues.info_pos), (id::TRACKS, cues.tracks_pos), (id::CUES, cues_pos)];
    for (target, pos) in targets {
        ebml::write_id(&mut sh, id::SEEK);
        let s_at = ebml::reserve_size(&mut sh);
        let s0 = sh.len();
        // SeekID's payload is the raw 4-octet element ID of the target master.
        ebml::write_binary(&mut sh, id::SEEK_ID, target);
        ebml::write_uint(&mut sh, id::SEEK_POSITION, pos);
        let s_len = (sh.len() - s0) as u64;
        ebml::patch_size(&mut sh, s_at, s_len);
    }
    let len = (sh.len() - body) as u64;
    ebml::patch_size(&mut sh, size_at, len);
    let rest = SEEKHEAD_RESERVE - sh.len();
    debug_assert!(rest >= 2, "reservation sized for the SeekHead plus a Void header");
    ebml::write_id(&mut sh, id::VOID);
    ebml::write_size(&mut sh, (rest - 2) as u64);
    sh.resize(SEEKHEAD_RESERVE, 0);
    sh
}

/// Scatter-mode muxer output (ZERO-COPY.md Stage 2): an ordered queue of **byte runs**
/// (the stream header, Cluster heads, SimpleBlock headers — the only bytes the muxer
/// itself produces) interleaved with **payload slices** (the refcounted [`Memory`] each
/// frame arrived in, forwarded without a copy). The consuming element drains pieces in
/// order — byte runs through pool slots (keeping the backpressure discipline), payloads
/// pushed as whole buffers — and calls [`reclaim`](Self::reclaim) once drained so the
/// byte arena's allocation is reused.
#[derive(Default)]
pub struct MuxOut {
    /// The shared byte arena every [`MuxPiece::Bytes`] range indexes into. Append-only
    /// while any piece is queued; cleared by [`reclaim`](Self::reclaim).
    bytes: Vec<u8>,
    /// The emission order.
    pieces: VecDeque<MuxPiece>,
    /// Total bytes ever queued (byte runs + payloads) — the absolute stream offset of
    /// whatever is queued next. What positions cue points and the SeekHead reservation
    /// (RFC 9559 §4 Segment Position) without the writer ever seeing the sink.
    emitted: u64,
}

/// One drainable piece of the muxed stream, in emission order.
pub enum MuxPiece {
    /// `MuxOut::header_bytes()[start..end]` — muxer-produced header bytes.
    Bytes { start: usize, end: usize },
    /// A frame's payload — forwarded by refcount, never copied.
    Payload(Memory),
}

impl MuxOut {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append header bytes by running `f` over the shared arena, queueing the written
    /// range as a piece (merged into a trailing contiguous byte run, so a Cluster head +
    /// its first block header drain as one run).
    fn append_bytes_with<R>(&mut self, f: impl FnOnce(&mut Vec<u8>) -> R) -> R {
        let start = self.bytes.len();
        let r = f(&mut self.bytes);
        let end = self.bytes.len();
        if end > start {
            self.emitted += (end - start) as u64;
            if let Some(MuxPiece::Bytes { end: e, .. }) = self.pieces.back_mut() {
                if *e == start {
                    *e = end;
                    return r;
                }
            }
            self.pieces.push_back(MuxPiece::Bytes { start, end });
        }
        r
    }

    /// Queue a payload piece (a refcount move). Empty payloads queue nothing — a
    /// zero-length frame is fully described by its SimpleBlock header.
    fn push_payload(&mut self, mem: Memory) {
        if !mem.is_empty() {
            self.emitted += mem.len() as u64;
            self.pieces.push_back(MuxPiece::Payload(mem));
        }
    }

    /// Total bytes ever queued — the absolute stream offset of whatever comes next.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The byte arena [`MuxPiece::Bytes`] ranges index into.
    pub fn header_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The next piece to drain, if any.
    pub fn front(&self) -> Option<&MuxPiece> {
        self.pieces.front()
    }

    /// The queued pieces in emission order — a drain's look-ahead (e.g. sizing one
    /// coalesced buffer over several consecutive small pieces before popping them).
    pub fn pieces(&self) -> impl Iterator<Item = &MuxPiece> {
        self.pieces.iter()
    }

    /// Remove and return the next piece.
    pub fn pop_front(&mut self) -> Option<MuxPiece> {
        self.pieces.pop_front()
    }

    /// `true` when nothing is queued.
    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Reuse the byte arena's allocation once every piece has drained (no-op while
    /// pieces still reference it).
    pub fn reclaim(&mut self) {
        if self.pieces.is_empty() {
            self.bytes.clear();
        }
    }

    /// Drop everything (a stop/reset — a fresh stream, so the offset restarts too).
    pub fn clear(&mut self) {
        self.pieces.clear();
        self.bytes.clear();
        self.emitted = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flac_track(n: u64) -> TrackConfig {
        TrackConfig::flac(n, vec![b'f', b'L', b'a', b'C', 0x80, 0, 0, 34], 48000.0, 2, 16)
    }

    /// Parse a VINT's total length from its first byte (RFC 8794 §4).
    fn vint_len(b: u8) -> usize {
        b.leading_zeros() as usize + 1
    }

    /// Parse an Element Data Size VINT at `*at`, advancing past it.
    fn parse_size(buf: &[u8], at: &mut usize) -> u64 {
        let n = vint_len(buf[*at]);
        let mask = if n >= 8 { 0 } else { 0xFFu8 >> n }; // n=8: marker byte holds no data bits
        let mut v = (buf[*at] & mask) as u64;
        for i in 1..n {
            v = (v << 8) | buf[*at + i] as u64;
        }
        *at += n;
        v
    }

    /// A big-endian EBML unsigned integer payload.
    fn parse_uint(buf: &[u8]) -> u64 {
        buf.iter().fold(0, |a, &b| (a << 8) | b as u64)
    }

    /// Cues round-trip (RFC 9559 §5.1.1, §5.1.5): with [`MatroskaWriter::enable_cues`]
    /// the Segment's first child is a SeekHead whose Seek entries land exactly on
    /// Info, Tracks and the end-of-file Cues; every CueClusterPosition lands on a
    /// Cluster ID whose Timestamp child equals the CueTime; only keyframe-bearing
    /// Clusters of the video track are cued.
    #[test]
    fn cues_and_seekhead_roundtrip() {
        let video = TrackConfig::video(1, "V_VP8", Vec::new(), 320, 240);
        let mut w = MatroskaWriter::new(vec![video, flac_track(2)]);
        w.enable_cues();
        let mut out = Vec::new();
        w.write_header(&mut out).unwrap();
        // Two clusters, each opened by a video (anchor-track) keyframe.
        w.write_frame(&mut out, 1, 0, &[1, 2, 3], true).unwrap();
        w.write_frame(&mut out, 2, 5_000_000, &[9], true).unwrap();
        w.write_frame(&mut out, 1, 40_000_000, &[4, 5], false).unwrap();
        w.write_frame(&mut out, 1, 80_000_000, &[6], true).unwrap();
        w.finalize(&mut out);

        // Segment data start: right after the Segment ID + its 1-octet 0xFF size.
        let seg = out.windows(4).position(|s| s == id::SEGMENT).expect("Segment");
        let seg_data = seg + 4 + 1;

        // First child: the patched SeekHead, then the Void filler.
        assert_eq!(&out[seg_data..seg_data + 4], id::SEEK_HEAD, "SeekHead is first");
        let mut at = seg_data + 4;
        let sh_len = parse_size(&out, &mut at) as usize;
        let sh_end = at + sh_len;
        let mut targets: Vec<(Vec<u8>, usize)> = Vec::new();
        while at < sh_end {
            assert_eq!(&out[at..at + 2], id::SEEK);
            at += 2;
            let len = parse_size(&out, &mut at) as usize;
            let end = at + len;
            assert_eq!(&out[at..at + 2], id::SEEK_ID);
            at += 2;
            let idlen = parse_size(&out, &mut at) as usize;
            let target = out[at..at + idlen].to_vec();
            at += idlen;
            assert_eq!(&out[at..at + 2], id::SEEK_POSITION);
            at += 2;
            let plen = parse_size(&out, &mut at) as usize;
            let pos = parse_uint(&out[at..at + plen]) as usize;
            at += plen;
            assert_eq!(at, end, "Seek entry consumed exactly");
            targets.push((target, seg_data + pos));
        }
        assert_eq!(out[sh_end], id::VOID[0], "Void filler follows the SeekHead");
        let find = |wanted: &[u8]| {
            targets.iter().find(|(t, _)| t == wanted).map(|&(_, p)| p).expect("seek target")
        };
        assert_eq!(&out[find(id::INFO)..][..4], id::INFO, "SeekHead → Info");
        assert_eq!(&out[find(id::TRACKS)..][..4], id::TRACKS, "SeekHead → Tracks");
        let cues_at = find(id::CUES);
        assert_eq!(&out[cues_at..][..4], id::CUES, "SeekHead → Cues");

        // Walk the Cues: each point lands on a Cluster whose Timestamp == CueTime.
        let mut at = cues_at + 4;
        let cues_len = parse_size(&out, &mut at) as usize;
        let cues_end = at + cues_len;
        let mut times = Vec::new();
        while at < cues_end {
            assert_eq!(&out[at..at + 1], id::CUE_POINT);
            at += 1;
            let plen = parse_size(&out, &mut at) as usize;
            let pend = at + plen;
            assert_eq!(&out[at..at + 1], id::CUE_TIME);
            at += 1;
            let tlen = parse_size(&out, &mut at) as usize;
            let time = parse_uint(&out[at..at + tlen]);
            at += tlen;
            assert_eq!(&out[at..at + 1], id::CUE_TRACK_POSITIONS);
            at += 1;
            let _tplen = parse_size(&out, &mut at);
            assert_eq!(&out[at..at + 1], id::CUE_TRACK);
            at += 1;
            let ctlen = parse_size(&out, &mut at) as usize;
            assert_eq!(parse_uint(&out[at..at + ctlen]), 1, "cues reference the video track");
            at += ctlen;
            assert_eq!(&out[at..at + 1], id::CUE_CLUSTER_POSITION);
            at += 1;
            let cplen = parse_size(&out, &mut at) as usize;
            let cpos = seg_data + parse_uint(&out[at..at + cplen]) as usize;
            at += cplen;
            assert_eq!(at, pend, "CuePoint consumed exactly");
            assert_eq!(&out[cpos..cpos + 4], id::CLUSTER, "cue lands on a Cluster ID");
            let mut cat = cpos + 4;
            let _csize = parse_size(&out, &mut cat);
            assert_eq!(&out[cat..cat + 1], id::TIMESTAMP);
            cat += 1;
            let tslen = parse_size(&out, &mut cat) as usize;
            assert_eq!(parse_uint(&out[cat..cat + tslen]), time, "CueTime == Cluster Timestamp");
            times.push(time);
        }
        assert_eq!(times, vec![0, 80], "both keyframe Clusters cued (1 ms ticks)");
    }

    /// The live-recording shape (RFC 9559 §5.1.2 Duration): duration unknown at header
    /// time → [`MatroskaWriter::reserve_duration`] emits a placeholder; after
    /// [`MatroskaWriter::finalize_scatter`] the caller applies the SeekHead patch AND
    /// the [`MatroskaWriter::duration_patch`] as positioned writes. The flattened
    /// stream must then parse with the patched duration and intact cues.
    #[test]
    fn reserved_duration_patches_in_scatter_stream() {
        use streamcraft_core::memory::Pool;
        let video = TrackConfig::video(1, "V_VP8", Vec::new(), 320, 240);
        let mut w = MatroskaWriter::new(vec![video]);
        w.enable_cues();
        w.reserve_duration();
        let pool = Pool::new(16);
        let mut out = MuxOut::new();
        w.write_header_scatter(&mut out).unwrap();
        for (ns, key) in [(0u64, true), (40_000_000, false), (2_000_000_000, true)] {
            let mut mem = pool.acquire_exact(3);
            mem.as_mut_full()[..3].copy_from_slice(&[7, 8, 9]);
            mem.set_len(3);
            w.write_frame_scatter(&mut out, 1, ns, mem, key).unwrap();
        }
        let sh_patch = w.finalize_scatter(&mut out).expect("cue-worthy clusters → patch");
        let dur_patch = w.duration_patch(2_040_000_000).expect("reserved → patch");

        // Flatten the scatter stream as a file sink would, then apply both patches.
        let mut file = Vec::new();
        while let Some(piece) = out.pop_front() {
            match piece {
                MuxPiece::Bytes { start, end } => file.extend_from_slice(&out.header_bytes()[start..end]),
                MuxPiece::Payload(mem) => file.extend_from_slice(mem.data()),
            }
        }
        for p in [&sh_patch, &dur_patch] {
            file[p.offset as usize..p.offset as usize + p.bytes.len()].copy_from_slice(&p.bytes);
        }

        let mut r = crate::MatroskaReader::new();
        r.push(&file).expect("patched stream parses");
        assert_eq!(r.duration_ns(), Some(2_040_000_000), "patched Duration read back");
        // The duration patch must not have clobbered its neighbours: the SeekHead's
        // Cues entry still resolves (the reader saw Tracks past the Info master).
        assert_eq!(r.tracks().len(), 1, "Tracks parse past the patched Info");
    }

    #[test]
    fn header_requires_tracks() {
        let mut w = MatroskaWriter::new(vec![]);
        let mut out = Vec::new();
        assert_eq!(w.write_header(&mut out), Err(WriteError::BadTracks));
    }

    #[test]
    fn duplicate_track_number_rejected() {
        let mut w = MatroskaWriter::new(vec![flac_track(1), flac_track(1)]);
        let mut out = Vec::new();
        assert_eq!(w.write_header(&mut out), Err(WriteError::BadTracks));
    }

    #[test]
    fn frame_before_header_rejected() {
        let mut w = MatroskaWriter::new(vec![flac_track(1)]);
        let mut out = Vec::new();
        assert_eq!(w.write_frame(&mut out, 1, 0, &[1, 2, 3], true), Err(WriteError::HeaderNotWritten));
    }

    #[test]
    fn frame_on_unknown_track_rejected() {
        let mut w = MatroskaWriter::new(vec![flac_track(1)]);
        let mut out = Vec::new();
        w.write_header(&mut out).unwrap();
        assert_eq!(w.write_frame(&mut out, 9, 0, &[1], true), Err(WriteError::UnknownTrack(9)));
    }

    /// The header begins with the exact EBML Header ID and DocType "matroska".
    #[test]
    fn header_starts_with_ebml_id_and_matroska_doctype() {
        let mut w = MatroskaWriter::new(vec![flac_track(1)]);
        let mut out = Vec::new();
        w.write_header(&mut out).unwrap();
        assert_eq!(&out[..4], id::EBML, "starts with the EBML Header ID 1A45DFA3");
        // DocType "matroska" appears verbatim somewhere in the header.
        assert!(
            out.windows(8).any(|w| w == b"matroska"),
            "DocType matroska present"
        );
        // The Segment ID follows the EBML Header, then the 0xFF unknown-size marker.
        let seg = out.windows(4).position(|w| w == id::SEGMENT).expect("Segment id");
        assert_eq!(out[seg + 4], 0xFF, "Segment opened with unknown size");
    }

    /// A SimpleBlock body decodes to the expected track / relative-ts / flags / frame.
    #[test]
    fn simple_block_body_layout() {
        let mut out = Vec::new();
        MatroskaWriter::write_simple_block(&mut out, 1, 123, &[0xAA, 0xBB, 0xCC], true);
        assert_eq!(&out[..1], id::SIMPLE_BLOCK); // 0xA3
        // size VINT, then body. Track 1 → one-octet VINT 0x81; ts 123 → 0x00 0x7B; flags
        // 0x80; frame AA BB CC. Body length = 1 + 2 + 1 + 3 = 7 → size VINT 0x87.
        assert_eq!(out[1], 0x87, "body length 7");
        assert_eq!(out[2], 0x81, "track number 1 as VINT");
        assert_eq!(&out[3..5], &123i16.to_be_bytes(), "relative timestamp big-endian");
        assert_eq!(out[5], 0x80, "keyframe flag");
        assert_eq!(&out[6..], &[0xAA, 0xBB, 0xCC], "frame verbatim");
    }

    #[test]
    fn negative_relative_timestamp_is_signed() {
        let mut out = Vec::new();
        MatroskaWriter::write_simple_block(&mut out, 2, -5, &[0x01], false);
        // Track 2 → VINT 0x82; ts -5 → 0xFFFB; flags 0x00.
        assert_eq!(out[2], 0x82);
        assert_eq!(&out[3..5], &(-5i16).to_be_bytes());
        assert_eq!(out[5], 0x00, "no keyframe flag");
    }

    /// A track number that needs a 2-octet VINT is length-accounted correctly.
    #[test]
    fn multibyte_track_number_body_length() {
        let mut out = Vec::new();
        // Track 200 needs a 2-octet VINT (200 > 126). Body = 2 + 2 + 1 + 1 = 6.
        MatroskaWriter::write_simple_block(&mut out, 200, 0, &[0x09], true);
        assert_eq!(out[1], 0x86, "body length 6 with 2-octet track VINT");
        // Track VINT 200 at width 2: 0x40 marker | 200 = 0x40C8.
        assert_eq!(&out[2..4], &[0x40, 0xC8]);
    }

    /// Successive small timestamps stay in one Cluster; a keyframe on the anchor track
    /// opens a new one. Clusters land in `out` when they *close* (staged so their real
    /// size is written — players reject unknown-size Clusters), so counts are observed
    /// at close boundaries and after finalize.
    #[test]
    fn anchor_keyframe_opens_new_cluster() {
        let mut w = MatroskaWriter::new(vec![flac_track(1)]);
        let mut out = Vec::new();
        w.write_header(&mut out).unwrap();
        w.write_frame(&mut out, 1, 0, &[1], true).unwrap(); // opens cluster #1 (staged)
        w.write_frame(&mut out, 1, 1_000_000, &[2], false).unwrap(); // same cluster (1 tick)
        assert_eq!(count_clusters(&out), 0, "open cluster is staged, not yet emitted");
        w.write_frame(&mut out, 1, 2_000_000, &[3], true).unwrap(); // keyframe → new cluster
        assert_eq!(count_clusters(&out), 1, "anchor keyframe closed+emitted cluster #1");
        w.finalize(&mut out);
        assert_eq!(count_clusters(&out), 2, "finalize emitted the last cluster");
    }

    /// A timestamp beyond the signed-16-bit window forces a new Cluster even without a
    /// keyframe.
    #[test]
    fn timestamp_overflow_opens_new_cluster() {
        let mut w = MatroskaWriter::new(vec![flac_track(1)]);
        let mut out = Vec::new();
        w.write_header(&mut out).unwrap();
        w.write_frame(&mut out, 1, 0, &[1], true).unwrap(); // cluster base tick 0
        // 40_000 ms ticks > 32767 → must open a new cluster (non-keyframe), closing and
        // emitting the first.
        w.write_frame(&mut out, 1, 40_000 * 1_000_000, &[2], false).unwrap();
        assert_eq!(count_clusters(&out), 1, "ts overflow closed+emitted the first cluster");
        w.finalize(&mut out);
        assert_eq!(count_clusters(&out), 2, "finalize emitted the overflow cluster");
    }

    /// Count Cluster IDs in a byte stream (a coarse structural probe for the tests above).
    fn count_clusters(out: &[u8]) -> usize {
        out.windows(4).filter(|w| *w == id::CLUSTER).count()
    }
}
