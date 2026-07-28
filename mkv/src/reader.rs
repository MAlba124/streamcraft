//! `MatroskaReader` — the demuxer's parse engine: streamed Matroska bytes in, decoded
//! tracks + per-track frames out (spec: `spec/MATROSKA.md`; RFC 9559, RFC 8794). This is the
//! read counterpart of [`MatroskaWriter`](crate::MatroskaWriter), written against the same
//! element ID tree and sizing rules; code cross-references them (`§ID-tree`, `§sizing`,
//! `§simpleblock`, RFC 9559 §10.3 for lacing).
//!
//! ## Incremental by construction (spec: latency #2)
//! Bytes arrive in arbitrary chunks that need not align to element boundaries, so the reader
//! **buffers internally** ([`push`](MatroskaReader::push)) and re-runs its parser each time,
//! consuming only whole elements and leaving a partial tail buffered for the next call. An
//! element split across an input boundary reports [`ReadError::Incomplete`] from the EBML
//! layer, which the reader treats as "wait for more", not an error. Consumed bytes are
//! compacted out of the buffer so steady-state memory is bounded by one Cluster's worth of
//! in-flight data, not the whole file.
//!
//! ## What it parses (the muxer's subset, plus the reader-only forms)
//! - **EBML Header** (RFC 8794 §11.2.4) — skipped as an opaque master (its version fields do
//!   not change how the Segment is read for the DocTypes we accept).
//! - **Segment** (unknown-size streamed master, §sizing) — descended into.
//! - **Info\TimestampScale** (§ID-tree) — the ns-per-tick used to turn block timestamps into
//!   nanoseconds; defaults to 1_000_000 if absent (the Matroska default).
//! - **Tracks\TrackEntry** — TrackNumber, CodecID, CodecPrivate, and Audio params, one
//!   [`Track`] per entry. Discovery ([`tracks`](MatroskaReader::tracks)) is complete once the
//!   Tracks master has been fully read.
//! - **Cluster\Timestamp** and **SimpleBlock** / **BlockGroup\Block** — decoded into
//!   [`Frame`]s with an absolute pts in nanoseconds, honouring all three lacing modes (RFC
//!   9559 §10.3): none, Xiph, EBML, and fixed-size.
//!
//! ## Untrusted input (spec: a crash on bad input is a P0)
//! Every field is bounds-checked and every arithmetic step is guarded; a truncated,
//! over-long, or structurally impossible element yields a [`ReadError`] the demuxer element
//! maps to a `streamcraft` error — the reader never panics or slices out of range.

use std::collections::VecDeque;

use crate::ebml::{self, id, ReadError};

/// A track discovered in the Tracks master (spec `§ID-tree` TrackEntry). Carries exactly what
/// a downstream decoder needs to start: the number frames are routed by, the CodecID, the
/// CodecPrivate init blob, and the audio params.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    /// `TrackNumber` (1-based) — the value SimpleBlock/Block frames carry to route to a track.
    pub track_number: u64,
    /// `CodecID`, e.g. `"A_FLAC"` (spec: A_FLAC mapping). Empty if the entry omitted it
    /// (malformed, but tolerated — the demuxer treats an unknown/empty id as a raw byte pad).
    pub codec_id: String,
    /// `CodecPrivate` — codec init data (FLAC `fLaC` + STREAMINFO for A_FLAC), empty if absent.
    pub codec_private: Vec<u8>,
    /// `Audio\SamplingFrequency` in Hz, or `0.0` if the entry carried no Audio master.
    pub sampling_frequency: f64,
    /// `Audio\Channels`, or `0` if absent.
    pub channels: u32,
    /// `Audio\BitDepth` (bits per sample), or `0` if absent.
    pub bit_depth: u32,
    /// `Video\PixelWidth` in pixels (RFC 9559 §5.1.4.1.28.6), or `0` if the entry carried no
    /// Video master. A downstream video decoder re-announces authoritative dimensions from the
    /// bitstream; this is the container's declared size, used to seed the pad announcement.
    pub pixel_width: u32,
    /// `Video\PixelHeight` in pixels (RFC 9559 §5.1.4.1.28.7), or `0` if absent.
    pub pixel_height: u32,
    /// `Colour\MatrixCoefficients` — an H.273 §8.3 code point. Initialized to the
    /// RFC 9559 default `2` ("unspecified") when the Colour element is absent — `0`
    /// cannot be the absent sentinel, it is a *valid* code point (identity/RGB).
    pub colour_matrix: u8,
    /// `Colour\Range` (RFC 9559: 1 limited, 2 full); default `0` = unspecified.
    pub colour_range: u8,
    /// `Colour\TransferCharacteristics` — H.273 §8.2 code point; default `2` = unspecified.
    pub colour_transfer: u8,
    /// `Colour\Primaries` — H.273 §8.1 code point; default `2` = unspecified.
    pub colour_primaries: u8,
}

/// One decoded frame ready to route to a track's src pad (spec `§simpleblock`, RFC 9559
/// §10.3). The `pts` is absolute, in nanoseconds (cluster base + block delta, both scaled by
/// TimestampScale). A laced block yields several `Frame`s sharing the block's timestamp (RFC
/// 9559 §10.3.5: the block timestamp applies to the first laced frame; later frames are
/// contiguous — this reader stamps them all with the block pts, which a fixed-rate audio
/// decoder refines from sample counts).
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// The track this frame belongs to (its `TrackNumber`).
    pub track_number: u64,
    /// Absolute presentation timestamp in nanoseconds.
    pub pts_ns: u64,
    /// Presentation duration in nanoseconds (`BlockGroup\BlockDuration` × TimestampScale),
    /// when the block declared one — `None` otherwise (RFC 9559 §5.1.3.6). Only a
    /// `BlockGroup` can carry it; a bare `SimpleBlock` never does, so a `SimpleBlock` frame
    /// is always `None`. This is load-bearing for **duration-bearing sparse tracks** —
    /// subtitles above all (S_TEXT/*), where the cue's on-screen span is `[pts, pts+dur)`
    /// and there is no next-frame delta to infer it from; the demuxer stamps it onto the
    /// output buffer's `duration` so the overlay has the interval.
    pub duration_ns: Option<u64>,
    /// True if the block flagged keyframe (SimpleBlock only; a BlockGroup Block with no
    /// ReferenceBlock is a keyframe, RFC 9559 §12.8 — see [`MatroskaReader`] for the rule).
    pub keyframe: bool,
    /// The frame payload — the codec bytes verbatim (one FLAC frame for A_FLAC).
    pub data: Vec<u8>,
}

/// The reader's high-level phase. Discovery (Header → Segment → Info → Tracks) must complete
/// before any frame is emitted; after it, the reader stays in `Streaming`, walking Clusters.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Reading top-level elements until the Tracks master has been fully parsed.
    Discovering,
    /// Tracks known; walking Clusters and decoding blocks.
    Streaming,
    /// Mid-Segment resume after a seek (spec: flush/seek): the byte source jumped to a new
    /// offset — either an exact cue-indexed Cluster start, or a proportional estimate that can
    /// land *anywhere*, including mid-block. Before parsing resumes we scan forward for the next
    /// Cluster ID (0x1F43B675), discarding any leading garbage; once found we drop back to
    /// [`Phase::Streaming`] at that Cluster's ID.
    Resyncing,
}

/// The default TimestampScale when Info omits it (Matroska `basics`: 1_000_000 ns/tick).
const DEFAULT_TIMESTAMP_SCALE: u64 = 1_000_000;

/// Incremental Matroska structure reader. Feed bytes with [`push`](Self::push); read the
/// discovered [`tracks`](Self::tracks) once [`tracks_ready`](Self::tracks_ready); drain
/// decoded [`Frame`]s with [`next_frame`](Self::next_frame). See the module docs for the
/// element subset and the incremental/robustness contract.
pub struct MatroskaReader {
    /// Bytes not yet fully consumed. The parser reads whole elements from the front and
    /// [`compact`](Self::compact)s consumed bytes out, so this holds at most one partial
    /// element plus the current Cluster's not-yet-parsed tail.
    buf: Vec<u8>,
    /// Offset into `buf` of the next unparsed byte at top level (or inside the Segment).
    pos: usize,
    phase: Phase,
    /// True once we have descended into the Segment (so top-level walking parses Segment
    /// children, not another EBML document).
    in_segment: bool,
    /// TimestampScale in ns/tick (Info\TimestampScale, or the default).
    timestamp_scale: u64,
    /// Tracks discovered from the Tracks master, in file order.
    tracks: Vec<Track>,
    /// True once the whole Tracks master has been parsed — discovery is complete.
    tracks_ready: bool,
    /// Presentation duration in ns (`Info\Duration` × TimestampScale), when declared.
    duration_ns: Option<u64>,
    /// The current Cluster's base timestamp in ticks (Cluster\Timestamp). `0` before the
    /// first Cluster.
    cluster_base_tick: i64,
    /// Decoded frames waiting to be drained by [`next_frame`](Self::next_frame), in stream
    /// order (a laced block pushes several at once).
    pending: VecDeque<Frame>,
    /// Recycled frame-payload buffers (streamcraft patch). [`push_frame`] draws a `Vec<u8>`
    /// from here instead of allocating; a consumer hands each drained frame's buffer back via
    /// [`recycle`](Self::recycle) once done with it, so steady-state framing allocates nothing.
    /// Bounded, so a stalled consumer cannot grow it without limit. Empty for consumers that
    /// don't recycle (probing, tests) — they simply keep allocating, which is still correct.
    free: Vec<Vec<u8>>,
    /// Reused scratch for a laced block's per-frame sizes: `decode_block` clears and refills
    /// this in place so a laced block (audio packing) allocates nothing per block. Unlaced
    /// blocks (the common video/one-frame-per-block case) never touch it.
    lace_sizes: Vec<usize>,
}

impl Default for MatroskaReader {
    fn default() -> Self {
        Self::new()
    }
}

impl MatroskaReader {
    /// A fresh reader awaiting the EBML Header.
    // COLD: reader construction — empty reusable buffers/collections, filled as bytes arrive.
    #[allow(clippy::disallowed_methods)]
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
            phase: Phase::Discovering,
            in_segment: false,
            timestamp_scale: DEFAULT_TIMESTAMP_SCALE,
            tracks: Vec::new(),
            tracks_ready: false,
            duration_ns: None,
            cluster_base_tick: 0,
            pending: VecDeque::new(),
            free: Vec::new(),
            lace_sizes: Vec::new(),
        }
    }

    /// Hand a drained frame's payload buffer back for reuse by a later [`push_frame`], so
    /// steady-state framing allocates nothing (streamcraft patch). Bounded — buffers past the
    /// cap are dropped. Recycling is optional: a consumer that never calls this just keeps
    /// allocating fresh buffers, with identical output.
    pub fn recycle(&mut self, buf: Vec<u8>) {
        const MAX_FREE: usize = 64;
        if self.free.len() < MAX_FREE {
            self.free.push(buf);
        }
    }

    /// The TimestampScale in ns/tick this stream declared (or the default until Info is read).
    pub fn timestamp_scale(&self) -> u64 {
        self.timestamp_scale
    }

    /// The presentation duration in ns, when the stream declared `Info\Duration`
    /// (RFC 9559 §5.1.2). `None` for duration-less (live-style) streams.
    pub fn duration_ns(&self) -> Option<u64> {
        self.duration_ns
    }

    /// The tracks discovered so far (complete once [`tracks_ready`](Self::tracks_ready)).
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// True once the entire Tracks master has been parsed — track discovery is final and it
    /// is safe to instantiate one src pad per [`tracks`](Self::tracks) entry.
    pub fn tracks_ready(&self) -> bool {
        self.tracks_ready
    }

    /// Feed `data` into the reader and parse as far as the buffered bytes allow (spec:
    /// incremental). Any decoded frames land in the queue drained by
    /// [`next_frame`](Self::next_frame). A structural error propagates; a merely-incomplete
    /// element is buffered for the next call (not an error).
    pub fn push(&mut self, data: &[u8]) -> Result<(), ReadError> {
        self.buf.extend_from_slice(data);
        self.parse()
    }

    /// Take the next decoded frame, if one is ready. Frames come out in stream order, already
    /// routed by [`track_number`](Frame::track_number) with an absolute nanosecond pts.
    pub fn next_frame(&mut self) -> Option<Frame> {
        self.pending.pop_front()
    }

    /// Drop the consumed prefix of `buf` so steady-state memory stays bounded (spec:
    /// performance #1 — no whole-file buffering). Only shifts when a worthwhile amount has
    /// been consumed, to keep the memmove rare.
    fn compact(&mut self) {
        // Shift once the consumed prefix is at least half the buffer (amortised O(1) per byte)
        // — the same bounded-tail strategy the FLAC stream decoder uses.
        if self.pos > 0 && (self.pos * 2 >= self.buf.len() || self.pos == self.buf.len()) {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
    }

    /// The main incremental parse loop. Reads whole top-level / Segment-level elements from
    /// `buf[pos..]`; stops (buffering the rest) on the first [`Incomplete`](ReadError::Incomplete).
    /// A definite-size master we care about (Info, Tracks, Cluster, BlockGroup) is only
    /// descended into once *fully* buffered, so its children are parsed from a contiguous
    /// window; the unknown-size Segment/Cluster are descended incrementally.
    fn parse(&mut self) -> Result<(), ReadError> {
        loop {
            match self.step() {
                Ok(true) => {
                    // Made progress; keep going and periodically reclaim consumed bytes.
                    self.compact();
                }
                Ok(false) => {
                    // No more whole elements buffered — wait for the next push.
                    self.compact();
                    return Ok(());
                }
                Err(ReadError::Incomplete) => {
                    // A partial element straddles the input boundary — buffer and retry later.
                    self.compact();
                    return Ok(());
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Parse one element at `buf[pos]` at the current nesting level. Returns `Ok(true)` if it
    /// consumed an element (advancing `pos`), `Ok(false)` if there is nothing more to do
    /// right now (buffer empty at this level), or an error. `Incomplete` bubbles up to
    /// [`parse`](Self::parse), which buffers.
    fn step(&mut self) -> Result<bool, ReadError> {
        if self.phase == Phase::Resyncing {
            return self.step_resync();
        }
        if self.pos >= self.buf.len() {
            return Ok(false);
        }
        let header = ebml::read_element_header(&self.buf, self.pos)?;
        // Stack-copy the (≤4-octet) EBML ID so the value outlives `header`'s borrow of
        // `self.buf` without a per-element heap `Vec` (this runs for *every* element — the
        // single hottest allocation site in the demux). EBML IDs are 1–4 octets (RFC 8794
        // §5, EBMLMaxIDLength default 4); a longer one is malformed here.
        let mut id_buf = [0u8; 4];
        let id_len = header.id.len();
        if id_len > id_buf.len() {
            return Err(ReadError::Malformed("EBML ID longer than 4 octets"));
        }
        id_buf[..id_len].copy_from_slice(header.id);
        let id = &id_buf[..id_len];
        let hdr_end = header.data_start;

        // Descend into the two streamed (unknown-size) masters incrementally: consume just
        // their header here, then continue parsing their children at the next level.
        if id == id::SEGMENT {
            self.pos = hdr_end;
            self.in_segment = true;
            return Ok(true);
        }
        if id == id::CLUSTER {
            // A Cluster resets its timestamp base; children follow (Timestamp, blocks).
            self.pos = hdr_end;
            self.cluster_base_tick = 0;
            if self.phase == Phase::Discovering && self.tracks_ready {
                self.phase = Phase::Streaming;
            }
            return Ok(true);
        }

        // Everything else is definite-size: we need its whole data buffered to act on it.
        let data_end = match header.data_end()? {
            Some(e) => e,
            None => {
                // An unknown-size element we do not special-case (an unexpected streamed
                // master) — reject rather than guess where it ends.
                return Err(ReadError::Malformed("unexpected unknown-size element"));
            }
        };
        if data_end > self.buf.len() {
            return Err(ReadError::Incomplete); // not all here yet — buffer more
        }
        let (ds, de) = (header.data_start, data_end);

        // The block decoders read `self.buf[ds..de]` and push into `self.pending`; to keep the
        // borrow checker happy while both are fields of `self`, decode straight from the range
        // (each helper re-slices `self.buf` and takes an explicit output queue). Info/Tracks
        // parse once per stream, so they slice a borrow locally with no `self` mutation until
        // after — captured here as owned indices.
        if id == id::INFO {
            let (scale, duration_ticks) = parse_info(&self.buf[ds..de])?;
            if let Some(scale) = scale {
                self.timestamp_scale = scale;
            }
            // Duration is declared in ticks; convert with the (possibly just-updated) scale.
            self.duration_ns =
                duration_ticks.map(|t| (t * self.timestamp_scale as f64).round() as u64);
        } else if id == id::TRACKS {
            let tracks = parse_tracks(&self.buf[ds..de])?;
            self.tracks.extend(tracks);
            self.tracks_ready = true;
        } else if id == id::TIMESTAMP {
            self.cluster_base_tick = read_uint(&self.buf[ds..de]) as i64;
        } else if id == id::SIMPLE_BLOCK {
            let (base, scale) = (self.cluster_base_tick, self.timestamp_scale);
            decode_simple_block(
                &self.buf[ds..de],
                base,
                scale,
                &mut self.pending,
                &mut self.free,
                &mut self.lace_sizes,
            )?;
        } else if id == id::BLOCK_GROUP {
            let (base, scale) = (self.cluster_base_tick, self.timestamp_scale);
            decode_block_group(
                &self.buf[ds..de],
                base,
                scale,
                &mut self.pending,
                &mut self.free,
                &mut self.lace_sizes,
            )?;
        }
        // Any other element (EBML Header, Cues, Tags, Void, unknown) is skipped as opaque.

        self.pos = de;
        Ok(true)
    }

    /// Scan forward for the next Cluster ID while [`Resyncing`](Phase::Resyncing) after a seek.
    /// The byte source jumped to a new offset that may fall mid-block (a proportional-estimate
    /// seek lands anywhere), so the buffered bytes at `pos` are not guaranteed to begin on an
    /// element boundary. We search `buf[pos..]` for the 4-octet Cluster ID (0x1F43B675) and, on
    /// finding it, drop every byte before it and switch back to [`Streaming`](Phase::Streaming)
    /// so the normal parser resumes at that Cluster.
    ///
    /// Returns `Ok(true)` when a Cluster ID was found and committed to (progress: garbage
    /// discarded, phase advanced), `Ok(false)` when none is in the buffer yet (need more bytes —
    /// the trailing window that might be a partial ID is retained). Never panics: bounds-checked
    /// throughout, and a **false positive** (four garbage bytes that happen to equal the Cluster
    /// ID but are not followed by a well-formed element size) resumes scanning past it rather
    /// than committing — so a genuine Cluster later in the buffer is still found, per the
    /// flush/seek contract ("worst case keep scanning or surface Incomplete", never crash).
    fn step_resync(&mut self) -> Result<bool, ReadError> {
        // Bounded memory (spec: performance #1 — no whole-file buffering): while scanning we
        // discard everything before the current search position, so the buffer holds only the
        // still-unscanned tail. Keep the last 3 bytes as a possible partial Cluster-ID prefix
        // that completes in the next push.
        const CID: &[u8] = id::CLUSTER; // [0x1F, 0x43, 0xB6, 0x75]
        // Scan from `pos`, testing each Cluster-ID candidate: a real Cluster is the ID followed
        // by a readable size VINT. A candidate whose size VINT is malformed is a false positive
        // — skip it and keep scanning. A candidate whose size VINT is not yet buffered is
        // Incomplete — wait for more bytes (do not discard it).
        let mut search_from = self.pos;
        loop {
            let hay = &self.buf[search_from..];
            let Some(rel) = find_subslice(hay, CID) else { break };
            let cand = search_from + rel;
            match ebml::read_element_header(&self.buf, cand) {
                Ok(_) => {
                    // A well-formed Cluster header: commit here and hand back to the normal parser.
                    self.pos = cand;
                    self.phase = Phase::Streaming;
                    self.cluster_base_tick = 0;
                    self.compact();
                    return Ok(true);
                }
                Err(ReadError::Incomplete) => {
                    // The candidate's size VINT straddles the input boundary — keep this
                    // candidate (drop only what precedes it) and wait for more bytes.
                    if cand > self.pos {
                        self.buf.drain(self.pos..cand);
                    }
                    return Ok(false);
                }
                Err(_) => {
                    // False positive (garbage matched the ID but is not a valid header) —
                    // advance one byte past this match and keep scanning.
                    search_from = cand + 1;
                }
            }
        }
        // No committable Cluster in the buffer. Drop everything except the trailing
        // (CID.len()-1) bytes — the longest prefix that could be the head of a Cluster ID split
        // across pushes — so scanning stays O(total bytes) and memory stays bounded. Drain the
        // consumed prefix (`..pos`) together with the fully-scanned middle in one go.
        let keep = CID.len() - 1;
        let end = self.buf.len();
        let drop_to = end.saturating_sub(keep);
        if drop_to > 0 {
            self.buf.drain(..drop_to);
        }
        self.pos = 0;
        Ok(false)
    }

    /// Reset the reader for a **mid-Segment resume** after a seek (spec: flush/seek), keeping the
    /// already-discovered stream shape. The byte source has repositioned to a new offset (an
    /// exact cue-indexed Cluster start, or a proportional estimate that can land mid-block), and
    /// the demuxer is about to feed bytes from there. We:
    /// - clear the buffered bytes, the parse cursor, and any half-decoded pending frames;
    /// - reset `cluster_base_tick` to 0 (the next Cluster's Timestamp re-establishes it — a
    ///   block's absolute pts is `cluster_base + rel_ts`, always recomputed per Cluster, so no
    ///   pre-seek base leaks into post-seek pts);
    /// - **keep** the discovered tracks, `tracks_ready`, `timestamp_scale`, and `duration_ns`
    ///   (those describe the whole stream and never change across a seek — re-discovering them
    ///   would need the header bytes again, which the post-seek source does not resend);
    /// - enter [`Resyncing`](Phase::Resyncing), which scans forward for the next Cluster ID
    ///   before resuming normal parsing, tolerating garbage before it.
    ///
    /// Unlike [`new`](Self::new) this preserves state a fresh reader would lose; unlike the
    /// EOS/stop path it does not tear down. The demuxer calls it on `Event::FlushStart`.
    pub fn resync_streaming(&mut self) {
        self.buf.clear();
        self.pos = 0;
        self.pending.clear();
        self.cluster_base_tick = 0;
        // Only resync from a resumable point if discovery already completed; if a seek somehow
        // arrives before the Tracks are known, stay in Discovering (the source will resend the
        // header from byte 0) rather than scan for a Cluster we cannot yet route.
        self.phase = if self.tracks_ready { Phase::Resyncing } else { Phase::Discovering };
    }

}

// The Info/Tracks/block decoders are **free functions** rather than methods so they can read
// a borrowed slice of the reader's internal buffer while the caller pushes decoded frames into
// the (separately-borrowed) pending queue — Rust's borrow checker rejects a `&mut self` method
// holding both a `&self.buf` slice and a `&mut self.pending`. Split-borrow via free functions
// keeps the parse zero-copy on the hot block path (no intermediate copy of the block bytes).

/// Parse the Segment `Info` master (spec `§ID-tree`): TimestampScale in ns/tick if present
/// and nonzero (a zero scale would divide by zero — the default is kept), and `Duration` in
/// **ticks** (RFC 9559 §5.1.2: a 4- or 8-octet float) if present. Other children
/// (MuxingApp, WritingApp, …) are skipped.
fn parse_info(data: &[u8]) -> Result<(Option<u64>, Option<f64>), ReadError> {
    let mut at = 0;
    let mut scale = None;
    let mut duration_ticks = None;
    while at < data.len() {
        let h = ebml::read_element_header(data, at)?;
        let end = h.data_end()?.ok_or(ReadError::Malformed("unknown-size child in Info"))?;
        if end > data.len() {
            return Err(ReadError::Malformed("Info child runs past its master"));
        }
        if h.id == id::TIMESTAMP_SCALE {
            let v = read_uint(&data[h.data_start..end]);
            if v != 0 {
                scale = Some(v);
            }
        } else if h.id == id::DURATION {
            let v = read_float(&data[h.data_start..end])?;
            if v.is_finite() && v > 0.0 {
                duration_ticks = Some(v);
            }
        }
        at = end;
    }
    Ok((scale, duration_ticks))
}

/// Parse the `Tracks` master into one [`Track`] per `TrackEntry` (spec `§ID-tree`).
// COLD: once per stream — Tracks discovery, owns the per-track config.
#[allow(clippy::disallowed_methods)]
fn parse_tracks(data: &[u8]) -> Result<Vec<Track>, ReadError> {
    let mut tracks = Vec::new();
    let mut at = 0;
    while at < data.len() {
        let h = ebml::read_element_header(data, at)?;
        let end = h.data_end()?.ok_or(ReadError::Malformed("unknown-size child in Tracks"))?;
        if end > data.len() {
            return Err(ReadError::Malformed("Tracks child runs past its master"));
        }
        if h.id == id::TRACK_ENTRY {
            tracks.push(parse_track_entry(&data[h.data_start..end])?);
        }
        at = end;
    }
    Ok(tracks)
}

/// Parse one `TrackEntry` master (spec `§ID-tree`) into a [`Track`]. Descends the nested
/// `Audio` master for the audio params and the `Video` master for pixel dimensions.
// COLD: once per track — owns the track's codec id / CodecPrivate.
#[allow(clippy::disallowed_methods)]
fn parse_track_entry(data: &[u8]) -> Result<Track, ReadError> {
    let mut track = Track {
        track_number: 0,
        codec_id: String::new(),
        codec_private: Vec::new(),
        sampling_frequency: 0.0,
        channels: 0,
        bit_depth: 0,
        pixel_width: 0,
        pixel_height: 0,
        // The RFC 9559 defaults: 2 = H.273 "unspecified" (0 is a *valid* matrix
        // code point — identity), 0 = unspecified range.
        colour_matrix: 2,
        colour_range: 0,
        colour_transfer: 2,
        colour_primaries: 2,
    };
    let mut at = 0;
    while at < data.len() {
        let h = ebml::read_element_header(data, at)?;
        let end = h.data_end()?.ok_or(ReadError::Malformed("unknown-size child in TrackEntry"))?;
        if end > data.len() {
            return Err(ReadError::Malformed("TrackEntry child runs past its master"));
        }
        let body = &data[h.data_start..end];
        if h.id == id::TRACK_NUMBER {
            track.track_number = read_uint(body);
        } else if h.id == id::CODEC_ID {
            track.codec_id = String::from_utf8_lossy(body).into_owned();
        } else if h.id == id::CODEC_PRIVATE {
            track.codec_private = body.to_vec();
        } else if h.id == id::AUDIO {
            parse_audio(body, &mut track)?;
        } else if h.id == id::VIDEO {
            parse_video(body, &mut track)?;
        }
        at = end;
    }
    if track.track_number == 0 {
        // A TrackEntry with no (or zero) TrackNumber cannot route blocks (RFC 9559
        // §5.1.4.1.1: TrackNumber MUST NOT be 0).
        return Err(ReadError::Malformed("TrackEntry with a zero/absent TrackNumber"));
    }
    Ok(track)
}

/// Parse the nested `Audio` master (spec `§ID-tree`) into the track's audio params.
/// SamplingFrequency is an EBML float, 4 or 8 octets (RFC 8794 §7.3).
fn parse_audio(data: &[u8], track: &mut Track) -> Result<(), ReadError> {
    let mut at = 0;
    while at < data.len() {
        let h = ebml::read_element_header(data, at)?;
        let end = h.data_end()?.ok_or(ReadError::Malformed("unknown-size child in Audio"))?;
        if end > data.len() {
            return Err(ReadError::Malformed("Audio child runs past its master"));
        }
        let body = &data[h.data_start..end];
        if h.id == id::SAMPLING_FREQUENCY {
            track.sampling_frequency = read_float(body)?;
        } else if h.id == id::CHANNELS {
            track.channels = read_uint(body) as u32;
        } else if h.id == id::BIT_DEPTH {
            track.bit_depth = read_uint(body) as u32;
        }
        at = end;
    }
    Ok(())
}

/// Parse the nested `Video` master (RFC 9559 §5.1.4.1.28) into the track's pixel dimensions.
/// PixelWidth/PixelHeight are EBML uints (RFC 8794 §7.1); other children (crop, colour, …)
/// are skipped — a video decoder re-announces authoritative dims from the bitstream.
fn parse_video(data: &[u8], track: &mut Track) -> Result<(), ReadError> {
    let mut at = 0;
    while at < data.len() {
        let h = ebml::read_element_header(data, at)?;
        let end = h.data_end()?.ok_or(ReadError::Malformed("unknown-size child in Video"))?;
        if end > data.len() {
            return Err(ReadError::Malformed("Video child runs past its master"));
        }
        let body = &data[h.data_start..end];
        if h.id == id::PIXEL_WIDTH {
            track.pixel_width = read_uint(body) as u32;
        } else if h.id == id::PIXEL_HEIGHT {
            track.pixel_height = read_uint(body) as u32;
        } else if h.id == id::COLOUR {
            parse_colour(body, track)?;
        }
        at = end;
    }
    Ok(())
}

/// Parse the nested `Colour` master (RFC 9559 §5.1.4.1.31) — the values are
/// ITU-T H.273 code points, stored raw here; the demuxer maps them to the
/// pipeline's categorical colorimetry names when announcing.
fn parse_colour(data: &[u8], track: &mut Track) -> Result<(), ReadError> {
    let mut at = 0;
    while at < data.len() {
        let h = ebml::read_element_header(data, at)?;
        let end = h.data_end()?.ok_or(ReadError::Malformed("unknown-size child in Colour"))?;
        if end > data.len() {
            return Err(ReadError::Malformed("Colour child runs past its master"));
        }
        let body = &data[h.data_start..end];
        if h.id == id::MATRIX_COEFFICIENTS {
            track.colour_matrix = read_uint(body) as u8;
        } else if h.id == id::COLOUR_RANGE {
            track.colour_range = read_uint(body) as u8;
        } else if h.id == id::TRANSFER_CHARACTERISTICS {
            track.colour_transfer = read_uint(body) as u8;
        } else if h.id == id::PRIMARIES {
            track.colour_primaries = read_uint(body) as u8;
        }
        at = end;
    }
    Ok(())
}

/// Decode a `BlockGroup` master (RFC 9559 §5.1.3.5): its single `Block` carries the frames
/// (same body layout as SimpleBlock but with no keyframe flag), optionally with a
/// `BlockDuration` (§5.1.3.6) and a `ReferenceBlock`. A Block with no `ReferenceBlock` is a
/// keyframe (RFC 9559 §12.8); a `ReferenceBlock` marks it as referencing another block, hence
/// not a keyframe. `BlockDuration` is in TimestampScale ticks — scaled to ns and attached to
/// every frame the block yields (the on-screen span for a subtitle cue). Pushes decoded
/// frames into `out`.
fn decode_block_group(
    data: &[u8],
    cluster_base_tick: i64,
    scale: u64,
    out: &mut VecDeque<Frame>,
    free: &mut Vec<Vec<u8>>,
    lace_buf: &mut Vec<usize>,
) -> Result<(), ReadError> {
    let mut at = 0;
    let mut has_reference = false;
    let mut block: Option<(usize, usize)> = None;
    let mut duration_ticks: Option<u64> = None;
    while at < data.len() {
        let h = ebml::read_element_header(data, at)?;
        let end = h.data_end()?.ok_or(ReadError::Malformed("unknown-size child in BlockGroup"))?;
        if end > data.len() {
            return Err(ReadError::Malformed("BlockGroup child runs past its master"));
        }
        if h.id == id::BLOCK {
            block = Some((h.data_start, end));
        } else if h.id == id::REFERENCE_BLOCK {
            has_reference = true;
        } else if h.id == id::BLOCK_DURATION {
            // Unsigned integer, in TimestampScale ticks (RFC 9559 §5.1.3.6).
            duration_ticks = Some(read_uint(&data[h.data_start..end]));
        }
        at = end;
    }
    if let Some((s, e)) = block {
        // Scale the tick duration to ns (saturating on adversarial input). Kept `None` when
        // the group declared none, so a downstream consumer can tell "unknown" from "zero".
        let duration_ns = duration_ticks.map(|t| t.saturating_mul(scale));
        // Keyframe status comes from the group (no ReferenceBlock => keyframe), not a flag in
        // the block body — pass it in.
        decode_block(&data[s..e], !has_reference, cluster_base_tick, scale, duration_ns, out, free, lace_buf)?;
    }
    Ok(())
}

/// Decode a **SimpleBlock** body (spec `§simpleblock`, RFC 9559 §10.2): track VINT, signed
/// 16-bit relative timestamp, a flags byte (keyframe bit `0x80`, lacing bits `0x06`), then the
/// lace. The keyframe flag is read from the flags byte. Pushes decoded frames into `out`.
fn decode_simple_block(
    data: &[u8],
    cluster_base_tick: i64,
    scale: u64,
    out: &mut VecDeque<Frame>,
    free: &mut Vec<Vec<u8>>,
    lace_buf: &mut Vec<usize>,
) -> Result<(), ReadError> {
    let (_track, _rel, flags, _start) = parse_block_header(data)?;
    let keyframe = flags & 0x80 != 0;
    // A SimpleBlock has no BlockDuration (only a BlockGroup carries one) — duration unknown.
    decode_block(data, keyframe, cluster_base_tick, scale, None, out, free, lace_buf)
}

/// Decode a Block/SimpleBlock body with an already-decided `keyframe` (from the flags byte for
/// a SimpleBlock, or from the enclosing BlockGroup's ReferenceBlock for a plain Block) and an
/// optional `duration_ns` (a BlockGroup's scaled BlockDuration; `None` for a SimpleBlock).
/// Parses the header, splits the lace, and pushes one [`Frame`] per laced frame into `out`. The
/// block duration applies to the block as a whole; every laced frame carries it (a laced
/// subtitle block is degenerate — the common single-cue case is unlaced anyway).
// Threads the reader's reused pool/scratch buffers (`free`, `lace_buf`) for zero-alloc decode.
#[allow(clippy::too_many_arguments)]
fn decode_block(
    data: &[u8],
    keyframe: bool,
    cluster_base_tick: i64,
    scale: u64,
    duration_ns: Option<u64>,
    out: &mut VecDeque<Frame>,
    free: &mut Vec<Vec<u8>>,
    lace_buf: &mut Vec<usize>,
) -> Result<(), ReadError> {
    let (track, rel_ts, flags, frames_start) = parse_block_header(data)?;
    let pts_ns = block_pts_ns(cluster_base_tick, rel_ts, scale);
    let lacing = (flags >> 1) & 0x03; // bits 0x06 → LACING (§10.1)
    let payload = data.get(frames_start..).ok_or(ReadError::Malformed("block payload truncated"))?;
    // The coded frame sizes fill `lace_buf` (reused, cleared here) so a laced block allocates
    // nothing; `header` is the octet count of lacing metadata before the frame data.
    lace_buf.clear();
    let header = match lacing {
        0b00 => {
            // No lacing: the whole payload is one frame (RFC 9559 §10.3.1).
            push_frame(out, free, track, pts_ns, duration_ns, keyframe, payload);
            return Ok(());
        }
        0b01 => xiph_lace_sizes(payload, lace_buf)?,
        0b11 => ebml_lace_sizes(payload, lace_buf)?,
        0b10 => fixed_lace_sizes(payload, lace_buf)?,
        _ => unreachable!("2-bit lacing field"),
    };
    // `header` bytes of lacing metadata precede the frames; carve each frame out.
    let mut off = header;
    for &sz in lace_buf.iter() {
        let end = off.checked_add(sz).ok_or(ReadError::Malformed("laced frame size overflow"))?;
        let frame = payload.get(off..end).ok_or(ReadError::Malformed("laced frame runs past the block"))?;
        push_frame(out, free, track, pts_ns, duration_ns, keyframe, frame);
        off = end;
    }
    // The last frame's size is whatever remains after the coded ones (§10.3.2/3/4).
    let last = payload.get(off..).ok_or(ReadError::Malformed("last laced frame runs past the block"))?;
    push_frame(out, free, track, pts_ns, duration_ns, keyframe, last);
    Ok(())
}

/// Parse the common Block/SimpleBlock header: `(track_number, rel_ts, flags, frames_start)`
/// where `frames_start` is the offset of the lacing data / first frame (RFC 9559 §10.1–2).
fn parse_block_header(data: &[u8]) -> Result<(u64, i16, u8, usize), ReadError> {
    let (track, track_len, _ao) = ebml::read_vint(data, 0)?;
    let after_track = track_len;
    // 16-bit signed relative timestamp, then a 1-octet flags byte.
    if after_track + 3 > data.len() {
        return Err(ReadError::Malformed("block header truncated (timestamp/flags)"));
    }
    let rel_ts = i16::from_be_bytes([data[after_track], data[after_track + 1]]);
    let flags = data[after_track + 2];
    Ok((track, rel_ts, flags, after_track + 3))
}

/// Absolute pts in ns for a block whose relative timestamp is `rel_ts` ticks: `(cluster base +
/// rel_ts) * TimestampScale`, clamped at 0 (a forward stream has no negative absolute time).
/// Saturating to avoid overflow on adversarial input.
fn block_pts_ns(cluster_base_tick: i64, rel_ts: i16, scale: u64) -> u64 {
    let abs_tick = cluster_base_tick.saturating_add(rel_ts as i64).max(0) as u64;
    abs_tick.saturating_mul(scale)
}

/// Take a recycled payload buffer that already has room for `needed` bytes, so the caller's
/// `extend_from_slice` never reallocates. The demux interleaves small (audio) and large
/// (video-keyframe) frames through one free-list, so a plain LIFO `pop()` hands a large frame a
/// small buffer and forces a grow every time; picking a buffer that already fits — else the
/// largest one (it grows once, then stays large and is recycled) — makes the steady state
/// realloc-free. Linear over a bounded (`MAX_FREE` = 64) list, so cheap.
// The `Vec::new()` fallback is an empty (zero-heap) buffer, hit only before the free-list has
// warmed; the actual copy allocation is the caller's `extend_from_slice`, which reuses a
// recycled buffer in steady state.
#[allow(clippy::disallowed_methods)]
fn take_fit(free: &mut Vec<Vec<u8>>, needed: usize) -> Vec<u8> {
    if let Some(i) = free.iter().position(|b| b.capacity() >= needed) {
        return free.swap_remove(i);
    }
    match free.iter().enumerate().max_by_key(|(_, b)| b.capacity()) {
        Some((i, _)) => free.swap_remove(i),
        None => Vec::new(),
    }
}

/// Queue one decoded frame, copying its bytes out of the shared buffer window into a payload
/// buffer drawn from `free` (a recycled buffer when the consumer returns them via
/// [`MatroskaReader::recycle`], else a fresh allocation).
fn push_frame(
    out: &mut VecDeque<Frame>,
    free: &mut Vec<Vec<u8>>,
    track: u64,
    pts_ns: u64,
    duration_ns: Option<u64>,
    keyframe: bool,
    data: &[u8],
) {
    let mut buf = take_fit(free, data.len());
    buf.clear();
    buf.extend_from_slice(data);
    out.push_back(Frame {
        track_number: track,
        pts_ns,
        duration_ns,
        keyframe,
        data: buf,
    });
}

/// Xiph lacing frame sizes (RFC 9559 §10.3.2): a 1-octet frame count minus one, then, for
/// every frame but the last, its size coded as a run of `0xFF` octets summed with a final
/// `< 0xFF` octet. Returns the coded sizes (all but the last frame) and the header length.
fn xiph_lace_sizes(payload: &[u8], sizes: &mut Vec<usize>) -> Result<usize, ReadError> {
    let count_minus_one = *payload.first().ok_or(ReadError::Malformed("Xiph lace: missing frame count"))?;
    let n_coded = count_minus_one as usize; // sizes stored for all but the last frame
    let mut at = 1usize;
    for _ in 0..n_coded {
        let mut sz = 0usize;
        loop {
            let b = *payload.get(at).ok_or(ReadError::Malformed("Xiph lace: size runs past the block"))?;
            at += 1;
            sz = sz.checked_add(b as usize).ok_or(ReadError::Malformed("Xiph lace: size overflow"))?;
            if b != 0xFF {
                break; // a value < 0xFF terminates this frame's size (§10.3.2)
            }
        }
        sizes.push(sz);
    }
    Ok(at)
}

/// EBML lacing frame sizes (RFC 9559 §10.3.3): a 1-octet frame count minus one, the first
/// frame size as an unsigned EBML VINT, then each subsequent size as a *signed* VINT delta
/// from the previous size (bias `2^((7*n)-1) - 1`). Fills `sizes` (all but the last frame) and
/// returns the header length.
fn ebml_lace_sizes(payload: &[u8], sizes: &mut Vec<usize>) -> Result<usize, ReadError> {
    let count_minus_one = *payload.first().ok_or(ReadError::Malformed("EBML lace: missing frame count"))?;
    let n_coded = count_minus_one as usize; // sizes stored for all but the last frame
    let mut at = 1usize;
    if n_coded == 0 {
        return Ok(at);
    }
    // First size: a plain unsigned VINT (its value is the frame length).
    let (first, len, _ao) = ebml::read_vint(payload, at)?;
    at += len;
    let mut prev = first as i64;
    sizes.push(usize::try_from(prev).map_err(|_| ReadError::Malformed("EBML lace: negative first size"))?);
    // Remaining sizes: signed deltas from the previous size.
    for _ in 1..n_coded {
        let (raw, len, _ao) = ebml::read_vint(payload, at)?;
        let bias = (1i64 << (7 * len - 1)) - 1; // §10.3.3: subtract 2^((7n)-1)-1
        let delta = raw as i64 - bias;
        prev = prev.checked_add(delta).ok_or(ReadError::Malformed("EBML lace: size delta overflow"))?;
        if prev < 0 {
            return Err(ReadError::Malformed("EBML lace: negative frame size"));
        }
        sizes.push(prev as usize);
        at += len;
    }
    Ok(at)
}

/// Fixed-size lacing frame sizes (RFC 9559 §10.3.4): a 1-octet frame count minus one; no sizes
/// are stored — every frame is `remaining / count` octets. Fills `sizes` for all but the last
/// frame (each the equal size) and returns the header length.
fn fixed_lace_sizes(payload: &[u8], sizes: &mut Vec<usize>) -> Result<usize, ReadError> {
    let count_minus_one = *payload.first().ok_or(ReadError::Malformed("fixed lace: missing frame count"))?;
    let count = count_minus_one as usize + 1;
    let body = payload.len() - 1; // frames follow the 1-octet header
    if body % count != 0 {
        return Err(ReadError::Malformed("fixed lace: block size not divisible by frame count"));
    }
    let each = body / count;
    // Sizes for all but the last frame; the caller derives the last from the remainder.
    sizes.extend(std::iter::repeat_n(each, count - 1));
    Ok(1)
}

/// Find the first occurrence of `needle` in `hay`, returning its start offset. A tiny
/// windowed scan — `needle` is a 4-octet element ID, so this is the byte-search the resync
/// uses to relocate the next Cluster (spec: flush/seek). `None` if absent.
fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// What a front SeekHead + Info yields a seeking caller (RFC 9559 §5.1.1): where the Cues
/// live and the geometry to turn Segment Positions into absolute byte offsets. Returned by
/// [`parse_seek_head`]. All positions are Segment Positions (relative to the first byte of the
/// Segment's data — RFC 9559 §4); add [`segment_data_start`](Self::segment_data_start) for an
/// absolute stream offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeekHeadInfo {
    /// Absolute stream offset of the Segment's data (the byte right after the Segment
    /// ID + size header). Every Segment Position is measured from here, so the Cues element
    /// lives at `segment_data_start + cues_pos` and a CueClusterPosition resolves the same way.
    pub segment_data_start: u64,
    /// The `Cues` element's Segment Position from the SeekHead, or `None` if the SeekHead
    /// had no Cues entry (an un-indexed file — the caller falls back to a proportional seek).
    pub cues_pos: Option<u64>,
    /// The stream's `Info\TimestampScale` in ns/tick (RFC 9559 §5.1.2.4), or the Matroska
    /// default (1_000_000) if Info omitted it. Needed to turn a CueTime (ticks) into ns.
    pub timestamp_scale: u64,
}

/// Walk the stream **header** bytes (EBML Header + the start of the Segment) and extract the
/// front SeekHead + TimestampScale (RFC 9559 §5.1.1) — the pure, IO-free parse the app's seek
/// index is built from. `header` must begin at byte 0 of the stream and include at least the
/// EBML Header, the Segment ID+size, the front SeekHead, and the Info master (i.e. the same
/// prefix `MkvDemux::new` is handed — everything up to the first Cluster).
///
/// Returns `None` only if the Segment itself cannot be located (not a Matroska stream, or the
/// header is truncated before the Segment header). A Segment with no SeekHead still returns
/// `Some` with `cues_pos == None` (an un-indexed file); the caller then seeks proportionally.
///
/// This does not read the file — the caller preads the Cues bytes at
/// `segment_data_start + cues_pos` and passes them to [`parse_cues`].
pub fn parse_seek_head(header: &[u8]) -> Option<SeekHeadInfo> {
    // Find the Segment header and the absolute offset of its data (RFC 9559 §4). We do a linear
    // top-level walk: the EBML Header is a definite-size master we skip past, then the Segment.
    let mut at = 0usize;
    let segment_data_start = loop {
        let h = ebml::read_element_header(header, at).ok()?;
        if h.id == id::SEGMENT {
            // The Segment is an unknown-size streamed master; its data starts right after the
            // ID + size header (the 0xFF marker).
            break h.data_start as u64;
        }
        // A definite-size top-level element (the EBML Header) — skip its whole data.
        let end = h.data_end().ok()??;
        if end <= at || end > header.len() {
            return None; // truncated before the Segment, or a malformed size
        }
        at = end;
    };

    // Walk the Segment's level-1 children within the header window, collecting the SeekHead's
    // Cues position and the Info TimestampScale. We stop at the first Cluster (frames start;
    // the front SeekHead and Info both precede it) or when the window runs out.
    let mut cues_pos = None;
    let mut timestamp_scale = DEFAULT_TIMESTAMP_SCALE;
    let mut at = segment_data_start as usize;
    while at < header.len() {
        let Ok(h) = ebml::read_element_header(header, at) else { break };
        if h.id == id::CLUSTER {
            break; // frames begin — nothing more of interest in the header
        }
        // Every level-1 child before the first Cluster (SeekHead, Info, Tracks, Void…) is
        // definite-size; a child whose data runs past the window means the caller passed too
        // little header — stop and return what we have.
        let Ok(Some(end)) = h.data_end() else { break };
        if end > header.len() {
            break;
        }
        if h.id == id::SEEK_HEAD {
            if let Some(p) = seek_head_cues_pos(&header[h.data_start..end]) {
                cues_pos = Some(p);
            }
        } else if h.id == id::INFO {
            if let Ok((Some(scale), _)) = parse_info(&header[h.data_start..end]) {
                timestamp_scale = scale;
            }
        }
        at = end;
    }

    Some(SeekHeadInfo { segment_data_start, cues_pos, timestamp_scale })
}

/// Scan one SeekHead master's children for the Seek entry whose SeekID targets the Cues
/// element, returning its SeekPosition (a Segment Position, RFC 9559 §5.1.1). `None` if the
/// SeekHead indexes no Cues. Malformed entries are skipped, never panicked on.
fn seek_head_cues_pos(data: &[u8]) -> Option<u64> {
    let mut at = 0usize;
    while at < data.len() {
        let h = ebml::read_element_header(data, at).ok()?;
        let end = h.data_end().ok()??;
        if end > data.len() {
            return None;
        }
        if h.id == id::SEEK {
            if let Some(p) = seek_entry_cues_pos(&data[h.data_start..end]) {
                return Some(p);
            }
        }
        at = end;
    }
    None
}

/// Parse one Seek entry (SeekID + SeekPosition): if its SeekID payload is the Cues element ID,
/// return its SeekPosition. `None` for any other target or a malformed entry.
fn seek_entry_cues_pos(data: &[u8]) -> Option<u64> {
    let mut at = 0usize;
    let mut is_cues = false;
    let mut pos = None;
    while at < data.len() {
        let h = ebml::read_element_header(data, at).ok()?;
        let end = h.data_end().ok()??;
        if end > data.len() {
            return None;
        }
        let body = &data[h.data_start..end];
        if h.id == id::SEEK_ID {
            // SeekID's payload is the raw element ID of the target master (the writer emits it
            // via `write_binary`). Compare against the Cues ID verbatim.
            is_cues = body == id::CUES;
        } else if h.id == id::SEEK_POSITION {
            pos = Some(read_uint(body));
        }
        at = end;
    }
    if is_cues { pos } else { None }
}

/// Parse a `Cues` master into a seek index: `(time_ns, absolute_byte)` per CuePoint, sorted
/// ascending by time, malformed entries skipped (RFC 9559 §5.1.5). `cues_bytes` is the Cues
/// element **including** its ID + size header — exactly the bytes the caller preads at
/// `segment_data_start + cues_pos` (the `SeekHeadInfo::cues_pos`), so a caller that read the
/// element header to learn its length can hand the same buffer straight back.
///
/// `segment_data_start` and `timestamp_scale` come from [`SeekHeadInfo`]. For each CuePoint:
/// `time_ns = CueTime × timestamp_scale`, and `absolute_byte = segment_data_start +
/// CueClusterPosition` (the position is a Segment Position, RFC 9559 §4). A CuePoint with
/// several CueTrackPositions (multi-track cues) contributes one entry per position; we take the
/// first per CuePoint (they share the CueTime and, for the writer's single-cue-track output,
/// there is exactly one). Never panics.
// COLD: once per stream — builds the seek index from the Cues element.
#[allow(clippy::disallowed_methods)]
pub fn parse_cues(
    cues_bytes: &[u8],
    segment_data_start: u64,
    timestamp_scale: u64,
) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    // Accept the element with its header (ID + size): find the Cues ID and descend into the
    // payload. If the header is missing (caller passed payload-only), a leading CuePoint ID
    // would not match Cues — so require the wrapping element for an unambiguous contract.
    let Ok(h) = ebml::read_element_header(cues_bytes, 0) else { return out };
    if h.id != id::CUES {
        return out;
    }
    let end = match h.data_end() {
        Ok(Some(e)) => e.min(cues_bytes.len()),
        _ => cues_bytes.len(),
    };
    let body = &cues_bytes[h.data_start..end];

    let mut at = 0usize;
    while at < body.len() {
        let Ok(cp) = ebml::read_element_header(body, at) else { break };
        let cp_end = match cp.data_end() {
            Ok(Some(e)) if e <= body.len() => e,
            _ => break, // truncated CuePoint — stop rather than mis-slice
        };
        if cp.id == id::CUE_POINT {
            if let Some((ticks, pos)) = parse_cue_point(&body[cp.data_start..cp_end]) {
                let time_ns = ticks.saturating_mul(timestamp_scale);
                let abs = segment_data_start.saturating_add(pos);
                out.push((time_ns, abs));
            }
        }
        at = cp_end;
    }
    out.sort_by_key(|&(t, _)| t);
    out
}

/// Parse one CuePoint (CueTime + the first CueTrackPositions' CueClusterPosition), returning
/// `(cue_time_ticks, cue_cluster_position)` (RFC 9559 §5.1.5). `None` if either field is
/// absent or the CuePoint is malformed — the caller skips it.
fn parse_cue_point(data: &[u8]) -> Option<(u64, u64)> {
    let mut at = 0usize;
    let mut time = None;
    let mut cluster_pos = None;
    while at < data.len() {
        let h = ebml::read_element_header(data, at).ok()?;
        let end = h.data_end().ok()??;
        if end > data.len() {
            return None;
        }
        if h.id == id::CUE_TIME {
            time = Some(read_uint(&data[h.data_start..end]));
        } else if h.id == id::CUE_TRACK_POSITIONS && cluster_pos.is_none() {
            // First CueTrackPositions wins (the writer emits one per CuePoint).
            cluster_pos = cue_track_positions_cluster(&data[h.data_start..end]);
        }
        at = end;
    }
    match (time, cluster_pos) {
        (Some(t), Some(p)) => Some((t, p)),
        _ => None,
    }
}

/// Parse one CueTrackPositions master for its CueClusterPosition (RFC 9559 §5.1.5.1.2), the
/// Segment Position of the Cluster the CuePoint indexes. `None` if absent/malformed.
fn cue_track_positions_cluster(data: &[u8]) -> Option<u64> {
    let mut at = 0usize;
    let mut pos = None;
    while at < data.len() {
        let h = ebml::read_element_header(data, at).ok()?;
        let end = h.data_end().ok()??;
        if end > data.len() {
            return None;
        }
        if h.id == id::CUE_CLUSTER_POSITION {
            pos = Some(read_uint(&data[h.data_start..end]));
        }
        at = end;
    }
    pos
}

/// Read an EBML unsigned integer from its big-endian octets (RFC 8794 §7.1). An empty slice
/// is `0` (a zero-length uint is legal). Saturates at `u64` for over-long input rather than
/// panicking.
fn read_uint(data: &[u8]) -> u64 {
    let mut v = 0u64;
    for &b in data.iter().take(8) {
        v = (v << 8) | b as u64;
    }
    v
}

/// Read an EBML float (RFC 8794 §7.3): a 4- or 8-octet big-endian IEEE 754 value. A
/// zero-length float is `0.0` (legal); any other length is malformed.
fn read_float(data: &[u8]) -> Result<f64, ReadError> {
    match data.len() {
        0 => Ok(0.0),
        4 => Ok(f32::from_bits(u32::from_be_bytes([data[0], data[1], data[2], data[3]])) as f64),
        8 => Ok(f64::from_bits(u64::from_be_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]))),
        _ => Err(ReadError::Malformed("EBML float must be 0, 4, or 8 octets")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{MatroskaWriter, TrackConfig};

    /// Build a minimal single-track MKV stream with the writer for the reader to consume.
    fn build_single(frames: &[&[u8]]) -> (Vec<u8>, Vec<u8>) {
        let codec_private = vec![b'f', b'L', b'a', b'C', 0x00, 0x00, 0x00, 0x22];
        let track = TrackConfig::flac(1, codec_private.clone(), 48_000.0, 2, 16);
        let mut w = MatroskaWriter::new(vec![track]);
        let mut out = Vec::new();
        w.write_header(&mut out).unwrap();
        let dur = 1_000_000u64; // 1 ms/frame in ns (1 tick at the default scale)
        for (i, f) in frames.iter().enumerate() {
            w.write_frame(&mut out, 1, i as u64 * dur, f, true).unwrap();
        }
        w.finalize(&mut out);
        (out, codec_private)
    }

    #[test]
    fn discovers_track_and_decodes_frames() {
        let frames: Vec<&[u8]> = vec![&[0xAA, 0xBB], &[0xCC], &[0xDD, 0xEE, 0xFF]];
        let (stream, codec_private) = build_single(&frames);

        let mut r = MatroskaReader::new();
        r.push(&stream).unwrap();
        assert!(r.tracks_ready(), "tracks discovered");
        assert_eq!(r.tracks().len(), 1);
        let t = &r.tracks()[0];
        assert_eq!(t.track_number, 1);
        assert_eq!(t.codec_id, "A_FLAC");
        assert_eq!(t.codec_private, codec_private);
        assert_eq!(t.sampling_frequency, 48_000.0);
        assert_eq!(t.channels, 2);
        assert_eq!(t.bit_depth, 16);

        let mut got = Vec::new();
        while let Some(f) = r.next_frame() {
            assert_eq!(f.track_number, 1);
            assert!(f.keyframe);
            got.push(f.data);
        }
        let want: Vec<Vec<u8>> = frames.iter().map(|f| f.to_vec()).collect();
        assert_eq!(got, want, "frames decode bit-exact and in order");
    }

    #[test]
    fn incremental_byte_by_byte_matches_whole() {
        let frames: Vec<&[u8]> = vec![&[1, 2, 3, 4], &[5, 6], &[7, 8, 9]];
        let (stream, _) = build_single(&frames);

        // Feed one byte at a time — the reader must buffer across every boundary.
        let mut r = MatroskaReader::new();
        for &b in &stream {
            r.push(&[b]).unwrap();
        }
        let mut got = Vec::new();
        while let Some(f) = r.next_frame() {
            got.push(f.data);
        }
        let want: Vec<Vec<u8>> = frames.iter().map(|f| f.to_vec()).collect();
        assert_eq!(got, want, "byte-by-byte feeding decodes identically");
    }

    #[test]
    fn timestamps_scale_to_nanoseconds() {
        // 3 frames spaced 1 tick (=1 ms) apart at the default scale.
        let frames: Vec<&[u8]> = vec![&[0], &[1], &[2]];
        let (stream, _) = build_single(&frames);
        let mut r = MatroskaReader::new();
        r.push(&stream).unwrap();
        let pts: Vec<u64> = std::iter::from_fn(|| r.next_frame().map(|f| f.pts_ns)).collect();
        assert_eq!(pts, vec![0, 1_000_000, 2_000_000], "pts in ns = tick * scale");
    }

    /// A subtitle-shaped stream: the writer emits SimpleBlocks, which carry no BlockDuration,
    /// so a SimpleBlock frame has `duration_ns == None` — the honest "unknown".
    #[test]
    fn simple_block_has_no_duration() {
        let frames: Vec<&[u8]> = vec![b"Hello", b"World"];
        let (stream, _) = build_single(&frames);
        let mut r = MatroskaReader::new();
        r.push(&stream).unwrap();
        while let Some(f) = r.next_frame() {
            assert_eq!(f.duration_ns, None, "a SimpleBlock never declares a duration");
        }
    }

    /// A `BlockGroup` with a `BlockDuration` (RFC 9559 §5.1.3.6) — the duration-bearing shape a
    /// subtitle (S_TEXT/*) track uses — parses to a `Frame` whose `duration_ns` is the scaled
    /// span. Hand-build a Cluster with one such group appended after a real header (the writer
    /// only emits SimpleBlocks, so BlockGroup is hand-rolled here).
    #[test]
    fn block_group_carries_scaled_block_duration() {
        use crate::ebml;
        // Header/Tracks from the writer (default 1 ms TimestampScale), no frames.
        let track = TrackConfig::flac(1, vec![b'f', b'L', b'a', b'C', 0, 0, 0, 0x22], 48_000.0, 2, 16);
        let mut w = MatroskaWriter::new(vec![track]);
        let mut stream = Vec::new();
        w.write_header(&mut stream).unwrap();

        // A Block body: track VINT (1 → 0x81), 16-bit signed rel timestamp, flags byte, payload.
        let cue_text = b"a subtitle cue";
        let mut block_body = Vec::new();
        block_body.push(0x81); // TrackNumber 1 as a 1-octet VINT
        block_body.extend_from_slice(&0i16.to_be_bytes()); // rel ts 0
        block_body.push(0x00); // flags: no keyframe/lacing (BlockGroup decides keyframe)
        block_body.extend_from_slice(cue_text);

        // BlockGroup { Block, BlockDuration = 1500 ticks }.
        let mut group = Vec::new();
        ebml::write_binary(&mut group, id::BLOCK, &block_body);
        ebml::write_uint(&mut group, id::BLOCK_DURATION, 1500);

        // Cluster { Timestamp = 2 ticks, BlockGroup }.
        let mut cluster_body = Vec::new();
        ebml::write_uint(&mut cluster_body, id::TIMESTAMP, 2);
        ebml::write_id(&mut cluster_body, id::BLOCK_GROUP);
        ebml::write_size(&mut cluster_body, group.len() as u64);
        cluster_body.extend_from_slice(&group);

        ebml::write_id(&mut stream, id::CLUSTER);
        ebml::write_size(&mut stream, cluster_body.len() as u64);
        stream.extend_from_slice(&cluster_body);

        let mut r = MatroskaReader::new();
        r.push(&stream).unwrap();
        let f = r.next_frame().expect("one cue frame");
        assert_eq!(f.track_number, 1);
        assert_eq!(f.data, cue_text);
        assert!(f.keyframe, "no ReferenceBlock → keyframe");
        // pts = (cluster base 2 + rel 0) * 1 ms; duration = 1500 * 1 ms.
        assert_eq!(f.pts_ns, 2_000_000, "pts scaled by TimestampScale");
        assert_eq!(f.duration_ns, Some(1_500_000_000), "BlockDuration scaled to ns");
        assert!(r.next_frame().is_none(), "exactly one frame");
    }

    #[test]
    fn truncated_input_never_panics() {
        let frames: Vec<&[u8]> = vec![&[0xDE, 0xAD, 0xBE, 0xEF]];
        let (stream, _) = build_single(&frames);
        // Every prefix must parse (buffering) without a panic or a spurious error.
        for n in 0..stream.len() {
            let mut r = MatroskaReader::new();
            let res = r.push(&stream[..n]);
            assert!(res.is_ok(), "prefix of length {n} must buffer, not error");
            // Draining what is ready must not panic either.
            while r.next_frame().is_some() {}
        }
    }

    // =================================================================================
    // Seek support (spec: flush/seek; RFC 9559 §5.1.1, §5.1.5): SeekHead/Cues parsing +
    // mid-stream resync. These build a real multi-cluster **video** stream with the writer
    // (cues on) so the seek index and Cluster offsets are the ones a player would see.
    // =================================================================================

    /// Build a multi-cluster V_VP8 video stream with cues enabled. Each `(ts_ns, keyframe,
    /// data)` is one frame on track 1; the writer opens a fresh Cluster on every anchor-track
    /// keyframe (spec `§simpleblock`), so a keyframe list of length K yields K cue-indexed
    /// Clusters. Returns the muxed stream bytes.
    fn build_cued_video(frames: &[(u64, bool, &[u8])]) -> Vec<u8> {
        let track = TrackConfig::vp8(1, 320, 240);
        let mut w = MatroskaWriter::new(vec![track]);
        w.enable_cues();
        let mut out = Vec::new();
        w.write_header(&mut out).unwrap();
        for &(ts, key, data) in frames {
            w.write_frame(&mut out, 1, ts, data, key).unwrap();
        }
        w.finalize(&mut out);
        out
    }

    /// The header prefix a seeking caller has on hand: everything up to the first Cluster
    /// (EBML Header + Segment header + SeekHead + Info + Tracks) — what `parse_seek_head`
    /// and `MkvDemux::new` consume.
    fn header_prefix(stream: &[u8]) -> &[u8] {
        let cl = stream.windows(4).position(|w| w == id::CLUSTER).expect("a Cluster");
        &stream[..cl]
    }

    /// Round-trip: mux a small multi-cluster cued file, then `parse_seek_head` + `parse_cues`
    /// on its bytes → the index entries point AT Cluster IDs, with ascending times matching the
    /// Cluster timestamps (RFC 9559 §5.1.5). Also checks the derived `segment_data_start` and
    /// TimestampScale.
    #[test]
    fn parse_seek_head_and_cues_roundtrip() {
        // Four anchor keyframes → four Clusters; the middle frame is a delta inside cluster #2.
        let stream = build_cued_video(&[
            (0, true, &[1, 2, 3]),
            (40_000_000, true, &[4, 5]),
            (60_000_000, false, &[6]), // delta — same cluster as the 40ms keyframe
            (80_000_000, true, &[7, 8, 9]),
            (120_000_000, true, &[10]),
        ]);

        let info = parse_seek_head(header_prefix(&stream)).expect("SeekHead + Info");
        assert_eq!(info.timestamp_scale, 1_000_000, "default 1 ms/tick");
        // Segment data start: right after the Segment ID + its 1-octet 0xFF size.
        let seg = stream.windows(4).position(|w| w == id::SEGMENT).unwrap();
        assert_eq!(info.segment_data_start, (seg + 4 + 1) as u64, "Segment data offset derived");
        let cues_pos = info.cues_pos.expect("SeekHead indexes Cues");

        // Pread the Cues element (id + size + payload) exactly, as the app would.
        let cues_abs = (info.segment_data_start + cues_pos) as usize;
        assert_eq!(&stream[cues_abs..cues_abs + 4], id::CUES, "Cues position lands on the Cues ID");
        let h = ebml::read_element_header(&stream, cues_abs).unwrap();
        let cues_end = h.data_end().unwrap().unwrap();
        let cues_bytes = &stream[cues_abs..cues_end];

        let index = parse_cues(cues_bytes, info.segment_data_start, info.timestamp_scale);
        // Four cue-worthy (keyframe) clusters at 0, 40, 80, 120 ms.
        assert_eq!(index.len(), 4, "one cue per keyframe cluster");
        let times: Vec<u64> = index.iter().map(|&(t, _)| t).collect();
        assert_eq!(times, vec![0, 40_000_000, 80_000_000, 120_000_000], "times ascending, in ns");
        assert!(times.windows(2).all(|w| w[0] <= w[1]), "sorted ascending");
        for &(_, abs) in &index {
            let a = abs as usize;
            assert_eq!(stream[a], 0x1F, "cue byte points AT a Cluster ID (0x1F...)");
            assert_eq!(&stream[a..a + 4], id::CLUSTER, "cue byte is the full Cluster ID");
        }
    }

    /// A file muxed **without** cues has no SeekHead → `cues_pos == None` (the app then seeks
    /// proportionally), yet `parse_seek_head` still returns the Segment geometry + scale.
    #[test]
    fn parse_seek_head_no_cues() {
        let (stream, _) = build_single(&[&[1], &[2], &[3]]);
        let info = parse_seek_head(header_prefix(&stream)).expect("Segment located");
        assert_eq!(info.cues_pos, None, "no SeekHead/Cues in an un-indexed file");
        assert_eq!(info.timestamp_scale, 1_000_000);
    }

    /// Malformed / non-Matroska bytes never panic and return `None` (no Segment).
    #[test]
    fn parse_seek_head_garbage_is_none() {
        assert_eq!(parse_seek_head(&[]), None);
        assert_eq!(parse_seek_head(&[0xFF; 32]), None);
        assert_eq!(parse_seek_head(&[0x1A, 0x45, 0xDF]), None, "truncated EBML id");
    }

    /// `parse_cues` skips a truncated / malformed CuePoint rather than panicking, and rejects
    /// bytes that are not a Cues element.
    #[test]
    fn parse_cues_robust_to_bad_bytes() {
        assert!(parse_cues(&[], 0, 1_000_000).is_empty());
        // A Cluster ID (not Cues) → empty, no panic.
        assert!(parse_cues(id::CLUSTER, 0, 1_000_000).is_empty());
        // A Cues element whose declared size runs past the buffer: clamp, decode nothing.
        let mut cues = Vec::new();
        ebml::write_id(&mut cues, id::CUES);
        ebml::write_size(&mut cues, 1000); // lies — no body follows
        assert!(parse_cues(&cues, 0, 1_000_000).is_empty());
    }

    /// Resync at an **exact** cue offset: after `resync_streaming`, feeding the reader bytes
    /// from a cue-indexed Cluster start decodes frames whose pts match that Cluster's timestamp
    /// (spec: flush/seek). The discovered track survives the resync.
    #[test]
    fn resync_at_exact_cluster_offset() {
        let stream = build_cued_video(&[
            (0, true, &[1, 2, 3]),
            (40_000_000, true, &[4, 5]),
            (80_000_000, true, &[6, 7]),
        ]);
        let info = parse_seek_head(header_prefix(&stream)).unwrap();
        let cues_abs = (info.segment_data_start + info.cues_pos.unwrap()) as usize;
        let h = ebml::read_element_header(&stream, cues_abs).unwrap();
        let cues_bytes = &stream[cues_abs..h.data_end().unwrap().unwrap()];
        let index = parse_cues(cues_bytes, info.segment_data_start, info.timestamp_scale);
        // Seek to the 40 ms cue (the second cluster).
        let (want_time, seek_byte) = index[1];
        assert_eq!(want_time, 40_000_000);

        // Discover tracks from the header, then resync and feed from the cue byte.
        let mut r = MatroskaReader::new();
        r.push(header_prefix(&stream)).unwrap();
        assert!(r.tracks_ready(), "tracks discovered before seek");
        while r.next_frame().is_some() {} // drain the header probe (no frames yet)

        r.resync_streaming();
        r.push(&stream[seek_byte as usize..]).unwrap();
        let first = r.next_frame().expect("a frame after resync");
        assert_eq!(first.pts_ns, 40_000_000, "first post-seek frame at the cue timestamp");
        assert_eq!(first.data, vec![4, 5], "the 40 ms cluster's frame, bit-exact");
        assert_eq!(r.tracks().len(), 1, "the discovered track survived resync");
    }

    /// Resync from a **garbage** offset (a few bytes before the cue, i.e. mid-block): the
    /// forward scan discards the garbage and recovers at the next Cluster — no panic, frames
    /// decode with the right pts (spec: flush/seek — a proportional estimate lands anywhere).
    #[test]
    fn resync_from_garbage_offset_recovers_at_next_cluster() {
        let stream = build_cued_video(&[
            (0, true, &[1, 2, 3]),
            (40_000_000, true, &[9, 9, 9, 9]),
            (80_000_000, true, &[7, 7]),
        ]);
        let info = parse_seek_head(header_prefix(&stream)).unwrap();
        let cues_abs = (info.segment_data_start + info.cues_pos.unwrap()) as usize;
        let h = ebml::read_element_header(&stream, cues_abs).unwrap();
        let cues_bytes = &stream[cues_abs..h.data_end().unwrap().unwrap()];
        let index = parse_cues(cues_bytes, info.segment_data_start, info.timestamp_scale);
        // Land 5 bytes *before* the 40 ms cue — squarely inside the previous cluster's data,
        // so the reader sees a partial block before the next Cluster ID.
        let cue_byte = index[1].1 as usize;
        let garbage_start = cue_byte - 5;

        let mut r = MatroskaReader::new();
        r.push(header_prefix(&stream)).unwrap();
        while r.next_frame().is_some() {}
        r.resync_streaming();
        r.push(&stream[garbage_start..]).unwrap();

        // The scan skips the 5 garbage bytes and any partial block, recovering at the 40 ms
        // Cluster — the first decoded frame is that Cluster's, not a mis-parse of the garbage.
        let first = r.next_frame().expect("recovered a frame");
        assert_eq!(first.pts_ns, 40_000_000, "recovered at the next Cluster's timestamp");
        assert_eq!(first.data, vec![9, 9, 9, 9]);
    }

    /// A **false-positive** Cluster ID in the garbage (four bytes equal to 0x1F43B675 but not
    /// followed by a valid element size) must be skipped, and the scan must recover at the real
    /// Cluster that follows (spec: flush/seek — worst case keep scanning, never crash).
    #[test]
    fn resync_skips_false_positive_cluster_id() {
        let stream = build_cued_video(&[(0, true, &[1]), (40_000_000, true, &[5, 5, 5])]);
        let info = parse_seek_head(header_prefix(&stream)).unwrap();
        let cues_abs = (info.segment_data_start + info.cues_pos.unwrap()) as usize;
        let h = ebml::read_element_header(&stream, cues_abs).unwrap();
        let cues_bytes = &stream[cues_abs..h.data_end().unwrap().unwrap()];
        let index = parse_cues(cues_bytes, info.segment_data_start, info.timestamp_scale);
        let cluster40 = index[1].1 as usize;

        let mut r = MatroskaReader::new();
        r.push(header_prefix(&stream)).unwrap();
        while r.next_frame().is_some() {}
        r.resync_streaming();
        // Feed: [garbage] [fake Cluster ID + 0x00 (an illegal size VINT — VintTooLong)] then the
        // real 40 ms Cluster onward. The scanner must reject the fake and land on the real one.
        let mut fed = vec![0xDE, 0xAD];
        fed.extend_from_slice(id::CLUSTER); // false positive
        fed.push(0x00); // 0x00 as a size VINT is illegal (width > 8) → not a valid header
        fed.extend_from_slice(&[0xBE, 0xEF]);
        fed.extend_from_slice(&stream[cluster40..]); // the real Cluster
        r.push(&fed).unwrap();

        let first = r.next_frame().expect("recovered past the false positive");
        assert_eq!(first.pts_ns, 40_000_000, "recovered at the real Cluster, not the fake ID");
        assert_eq!(first.data, vec![5, 5, 5]);
    }

    /// Resync byte-by-byte from a garbage offset: the scan must tolerate a Cluster ID split
    /// across pushes and never panic (spec: incremental + flush/seek).
    #[test]
    fn resync_scan_survives_split_cluster_id() {
        let stream = build_cued_video(&[
            (0, true, &[1]),
            (40_000_000, true, &[2, 2]),
        ]);
        let info = parse_seek_head(header_prefix(&stream)).unwrap();
        let cues_abs = (info.segment_data_start + info.cues_pos.unwrap()) as usize;
        let h = ebml::read_element_header(&stream, cues_abs).unwrap();
        let cues_bytes = &stream[cues_abs..h.data_end().unwrap().unwrap()];
        let index = parse_cues(cues_bytes, info.segment_data_start, info.timestamp_scale);
        let start = index[1].1 as usize - 7; // a few bytes before the cue

        let mut r = MatroskaReader::new();
        r.push(header_prefix(&stream)).unwrap();
        while r.next_frame().is_some() {}
        r.resync_streaming();
        for &b in &stream[start..] {
            r.push(&[b]).expect("byte-by-byte resync never errors");
        }
        let first = r.next_frame().expect("recovered a frame");
        assert_eq!(first.pts_ns, 40_000_000);
        assert_eq!(first.data, vec![2, 2]);
    }
}
