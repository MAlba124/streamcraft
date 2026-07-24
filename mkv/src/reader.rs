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
    /// The current Cluster's base timestamp in ticks (Cluster\Timestamp). `0` before the
    /// first Cluster.
    cluster_base_tick: i64,
    /// Decoded frames waiting to be drained by [`next_frame`](Self::next_frame), in stream
    /// order (a laced block pushes several at once).
    pending: VecDeque<Frame>,
}

impl Default for MatroskaReader {
    fn default() -> Self {
        Self::new()
    }
}

impl MatroskaReader {
    /// A fresh reader awaiting the EBML Header.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
            phase: Phase::Discovering,
            in_segment: false,
            timestamp_scale: DEFAULT_TIMESTAMP_SCALE,
            tracks: Vec::new(),
            tracks_ready: false,
            cluster_base_tick: 0,
            pending: VecDeque::new(),
        }
    }

    /// The TimestampScale in ns/tick this stream declared (or the default until Info is read).
    pub fn timestamp_scale(&self) -> u64 {
        self.timestamp_scale
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
        if self.pos >= self.buf.len() {
            return Ok(false);
        }
        let header = ebml::read_element_header(&self.buf, self.pos)?;
        let id = header.id.to_vec();
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
            let scale = parse_info(&self.buf[ds..de])?;
            if let Some(scale) = scale {
                self.timestamp_scale = scale;
            }
        } else if id == id::TRACKS {
            let tracks = parse_tracks(&self.buf[ds..de])?;
            self.tracks.extend(tracks);
            self.tracks_ready = true;
        } else if id == id::TIMESTAMP {
            self.cluster_base_tick = read_uint(&self.buf[ds..de]) as i64;
        } else if id == id::SIMPLE_BLOCK {
            let (base, scale) = (self.cluster_base_tick, self.timestamp_scale);
            decode_simple_block(&self.buf[ds..de], base, scale, &mut self.pending)?;
        } else if id == id::BLOCK_GROUP {
            let (base, scale) = (self.cluster_base_tick, self.timestamp_scale);
            decode_block_group(&self.buf[ds..de], base, scale, &mut self.pending)?;
        }
        // Any other element (EBML Header, Cues, Tags, Void, unknown) is skipped as opaque.

        self.pos = de;
        Ok(true)
    }

}

// The Info/Tracks/block decoders are **free functions** rather than methods so they can read
// a borrowed slice of the reader's internal buffer while the caller pushes decoded frames into
// the (separately-borrowed) pending queue — Rust's borrow checker rejects a `&mut self` method
// holding both a `&self.buf` slice and a `&mut self.pending`. Split-borrow via free functions
// keeps the parse zero-copy on the hot block path (no intermediate copy of the block bytes).

/// Parse the Segment `Info` master for TimestampScale (spec `§ID-tree`). Returns the scale in
/// ns/tick if present and nonzero (a zero scale would divide by zero — the default is kept);
/// other children (MuxingApp, WritingApp, Duration, …) are skipped.
fn parse_info(data: &[u8]) -> Result<Option<u64>, ReadError> {
    let mut at = 0;
    let mut scale = None;
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
        }
        at = end;
    }
    Ok(scale)
}

/// Parse the `Tracks` master into one [`Track`] per `TrackEntry` (spec `§ID-tree`).
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
/// `Audio` master for the audio params.
fn parse_track_entry(data: &[u8]) -> Result<Track, ReadError> {
    let mut track = Track {
        track_number: 0,
        codec_id: String::new(),
        codec_private: Vec::new(),
        sampling_frequency: 0.0,
        channels: 0,
        bit_depth: 0,
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

/// Decode a `BlockGroup` master (RFC 9559 §5.1.3.5): its single `Block` carries the frames
/// (same body layout as SimpleBlock but with no keyframe flag). A Block with no
/// `ReferenceBlock` is a keyframe (RFC 9559 §12.8); a `ReferenceBlock` marks it as referencing
/// another block, hence not a keyframe. Pushes decoded frames into `out`.
fn decode_block_group(
    data: &[u8],
    cluster_base_tick: i64,
    scale: u64,
    out: &mut VecDeque<Frame>,
) -> Result<(), ReadError> {
    let mut at = 0;
    let mut has_reference = false;
    let mut block: Option<(usize, usize)> = None;
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
        }
        at = end;
    }
    if let Some((s, e)) = block {
        // Keyframe status comes from the group (no ReferenceBlock => keyframe), not a flag in
        // the block body — pass it in.
        decode_block(&data[s..e], !has_reference, cluster_base_tick, scale, out)?;
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
) -> Result<(), ReadError> {
    let (_track, _rel, flags, _start) = parse_block_header(data)?;
    let keyframe = flags & 0x80 != 0;
    decode_block(data, keyframe, cluster_base_tick, scale, out)
}

/// Decode a Block/SimpleBlock body with an already-decided `keyframe` (from the flags byte for
/// a SimpleBlock, or from the enclosing BlockGroup's ReferenceBlock for a plain Block). Parses
/// the header, splits the lace, and pushes one [`Frame`] per laced frame into `out`.
fn decode_block(
    data: &[u8],
    keyframe: bool,
    cluster_base_tick: i64,
    scale: u64,
    out: &mut VecDeque<Frame>,
) -> Result<(), ReadError> {
    let (track, rel_ts, flags, frames_start) = parse_block_header(data)?;
    let pts_ns = block_pts_ns(cluster_base_tick, rel_ts, scale);
    let lacing = (flags >> 1) & 0x03; // bits 0x06 → LACING (§10.1)
    let payload = data.get(frames_start..).ok_or(ReadError::Malformed("block payload truncated"))?;
    let sizes = match lacing {
        0b00 => {
            // No lacing: the whole payload is one frame (RFC 9559 §10.3.1).
            push_frame(out, track, pts_ns, keyframe, payload);
            return Ok(());
        }
        0b01 => xiph_lace_sizes(payload)?,
        0b11 => ebml_lace_sizes(payload)?,
        0b10 => fixed_lace_sizes(payload)?,
        _ => unreachable!("2-bit lacing field"),
    };
    // `sizes.header` bytes of lacing metadata precede the frames; carve each frame out.
    let mut off = sizes.header;
    for &sz in &sizes.frames {
        let end = off.checked_add(sz).ok_or(ReadError::Malformed("laced frame size overflow"))?;
        let frame = payload.get(off..end).ok_or(ReadError::Malformed("laced frame runs past the block"))?;
        push_frame(out, track, pts_ns, keyframe, frame);
        off = end;
    }
    // The last frame's size is whatever remains after the coded ones (§10.3.2/3/4).
    let last = payload.get(off..).ok_or(ReadError::Malformed("last laced frame runs past the block"))?;
    push_frame(out, track, pts_ns, keyframe, last);
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

/// Queue one decoded frame (copying its bytes out of the shared buffer window into `out`).
fn push_frame(out: &mut VecDeque<Frame>, track: u64, pts_ns: u64, keyframe: bool, data: &[u8]) {
    out.push_back(Frame {
        track_number: track,
        pts_ns,
        keyframe,
        data: data.to_vec(),
    });
}

/// Xiph lacing frame sizes (RFC 9559 §10.3.2): a 1-octet frame count minus one, then, for
/// every frame but the last, its size coded as a run of `0xFF` octets summed with a final
/// `< 0xFF` octet. Returns the coded sizes (all but the last frame) and the header length.
fn xiph_lace_sizes(payload: &[u8]) -> Result<LaceSizes, ReadError> {
    let count_minus_one = *payload.first().ok_or(ReadError::Malformed("Xiph lace: missing frame count"))?;
    let n_coded = count_minus_one as usize; // sizes stored for all but the last frame
    let mut sizes = Vec::with_capacity(n_coded);
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
    Ok(LaceSizes { header: at, frames: sizes })
}

/// EBML lacing frame sizes (RFC 9559 §10.3.3): a 1-octet frame count minus one, the first
/// frame size as an unsigned EBML VINT, then each subsequent size as a *signed* VINT delta
/// from the previous size (bias `2^((7*n)-1) - 1`). Returns the coded sizes and header length.
fn ebml_lace_sizes(payload: &[u8]) -> Result<LaceSizes, ReadError> {
    let count_minus_one = *payload.first().ok_or(ReadError::Malformed("EBML lace: missing frame count"))?;
    let n_coded = count_minus_one as usize; // sizes stored for all but the last frame
    let mut sizes = Vec::with_capacity(n_coded);
    let mut at = 1usize;
    if n_coded == 0 {
        return Ok(LaceSizes { header: at, frames: sizes });
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
    Ok(LaceSizes { header: at, frames: sizes })
}

/// Fixed-size lacing frame sizes (RFC 9559 §10.3.4): a 1-octet frame count minus one; no sizes
/// are stored — every frame is `remaining / count` octets. Returns the sizes for all but the
/// last frame (each the equal size) and the header length.
fn fixed_lace_sizes(payload: &[u8]) -> Result<LaceSizes, ReadError> {
    let count_minus_one = *payload.first().ok_or(ReadError::Malformed("fixed lace: missing frame count"))?;
    let count = count_minus_one as usize + 1;
    let body = payload.len() - 1; // frames follow the 1-octet header
    if body % count != 0 {
        return Err(ReadError::Malformed("fixed lace: block size not divisible by frame count"));
    }
    let each = body / count;
    // Sizes for all but the last frame; the caller derives the last from the remainder.
    Ok(LaceSizes { header: 1, frames: vec![each; count - 1] })
}

/// The decoded lacing layout: how many header octets precede the frame data, and the sizes of
/// every frame *except the last* (whose size the caller derives from the block remainder).
struct LaceSizes {
    /// Number of octets from the start of the block payload to the first frame's data.
    header: usize,
    /// Sizes of all frames but the last, in order.
    frames: Vec<usize>,
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
}
