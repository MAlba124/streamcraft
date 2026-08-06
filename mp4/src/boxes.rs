//! Hand-written ISO base media file format (ISO-BMFF) box grammar (spec: ISO/IEC
//! 14496-12; `mp4/NOTES.md` cites the edition). Every media MP4 is a tree of **boxes**:
//! a `size` + `type` header, then either child boxes (a container) or a typed payload
//! (a leaf). This module is the read side only — the header grammar, container
//! recursion, and the leaf boxes the sample-table resolver needs (§8.5–§8.7).
//!
//! ## Box header (§4.2)
//! ```text
//! size   u32   total box length incl. this header; 0 = "to end of file",
//!              1 = "64-bit size follows"
//! type   u32   the four-character code (FourCC), e.g. `moov`, `stbl`
//! [ largesize u64 ]        present iff size == 1 (§4.2)
//! [ usertype u8[16] ]      present iff type == `uuid` (§4.2) — skipped
//! ```
//! A **FullBox** (§4.2) additionally begins its payload with a `version` u8 and a
//! 24-bit `flags`; the leaf parsers that need it read those first.
//!
//! ## Untrusted input (spec: "a crash on bad input is a P0")
//! Every field is bounds-checked and every declared length is validated against the
//! remaining window; a truncated or structurally impossible box yields a [`BoxError`],
//! never a panic or an out-of-range slice. The demuxer maps such an error to a loud
//! `profluens` error.
//!
//! ## Writer note (a muxer is a separate follow-up)
//! The box grammar is symmetric — a muxer would write the same headers this reader
//! parses. The [`fourcc`] helpers and the [`BoxHeader`] shape are deliberately
//! write-agnostic so a `writer.rs` can join later without reshaping the read side (see
//! the crate docs, "Not yet").

// COLD: pure box-parse grammar — runs once per stream at preroll, building owned sample
// tables that outlive process(); nothing here is on the per-sample hot path.
#![allow(clippy::disallowed_methods)]

/// A four-character box type code (FourCC, §4.2), stored as its raw big-endian octets so
/// it compares directly against the [`FourCc`] constants (no per-compare string alloc).
pub type FourCc = [u8; 4];

/// Build a [`FourCc`] from a 4-byte ASCII literal (compile-time). Used for the box-type
/// constants below; a shared helper so a future muxer can name boxes the same way.
pub const fn fourcc(s: &[u8; 4]) -> FourCc {
    [s[0], s[1], s[2], s[3]]
}

/// The box types the demuxer recognises (ISO/IEC 14496-12 section per group). Everything
/// not listed is skipped as opaque (§4.2: a reader ignores boxes it does not understand).
pub mod boxtype {
    use super::{fourcc, FourCc};

    // File / movie structure (§4.3, §8.1, §8.2, §8.3).
    pub const FTYP: FourCc = fourcc(b"ftyp"); // §4.3 File Type
    pub const MOOV: FourCc = fourcc(b"moov"); // §8.2.1 Movie
    pub const MVHD: FourCc = fourcc(b"mvhd"); // §8.2.2 Movie Header
    pub const TRAK: FourCc = fourcc(b"trak"); // §8.3.1 Track
    pub const TKHD: FourCc = fourcc(b"tkhd"); // §8.3.2 Track Header
    pub const EDTS: FourCc = fourcc(b"edts"); // §8.6.5 Edit
    pub const ELST: FourCc = fourcc(b"elst"); // §8.6.6 Edit List
    pub const MDIA: FourCc = fourcc(b"mdia"); // §8.4.1 Media
    pub const MDHD: FourCc = fourcc(b"mdhd"); // §8.4.2 Media Header
    pub const HDLR: FourCc = fourcc(b"hdlr"); // §8.4.3 Handler Reference
    pub const MINF: FourCc = fourcc(b"minf"); // §8.4.4 Media Information
    pub const STBL: FourCc = fourcc(b"stbl"); // §8.5.1 Sample Table

    // Sample table children (§8.5–§8.7).
    pub const STSD: FourCc = fourcc(b"stsd"); // §8.5.2 Sample Description
    pub const STTS: FourCc = fourcc(b"stts"); // §8.6.1.2 Decoding Time to Sample
    pub const CTTS: FourCc = fourcc(b"ctts"); // §8.6.1.3 Composition Time to Sample
    pub const STSZ: FourCc = fourcc(b"stsz"); // §8.7.3.2 Sample Size
    pub const STZ2: FourCc = fourcc(b"stz2"); // §8.7.3.3 Compact Sample Size
    pub const STSC: FourCc = fourcc(b"stsc"); // §8.7.4 Sample to Chunk
    pub const STCO: FourCc = fourcc(b"stco"); // §8.7.5 Chunk Offset (32-bit)
    pub const CO64: FourCc = fourcc(b"co64"); // §8.7.5 Chunk Offset (64-bit)
    pub const STSS: FourCc = fourcc(b"stss"); // §8.6.2 Sync Sample

    // Fragmented-movie boxes — DETECTED so v1 can error loudly (§8.8; out of scope).
    pub const MVEX: FourCc = fourcc(b"mvex"); // §8.8.1 Movie Extends
    pub const MOOF: FourCc = fourcc(b"moof"); // §8.8.4 Movie Fragment

    // Media data (§8.1.1) — the byte pool samples are sliced out of.
    pub const MDAT: FourCc = fourcc(b"mdat"); // §8.1.1 Media Data
}

/// A parse failure. The demuxer maps this to a `profluens` error; there is no
/// "incomplete" variant here because the box layer always parses a fully-buffered header
/// slice (the constructor-supplied head, or a `stbl` already buffered by the reader) — the
/// streaming/cross-boundary buffering lives one layer up in [`crate::reader`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BoxError {
    /// A box or field ran out of bytes before its declared length (truncated input).
    Truncated(&'static str),
    /// A structurally impossible value: a size smaller than its own header, a child that
    /// runs past its parent, a length that overflows `usize`, a version we do not support.
    Malformed(&'static str),
}

/// One parsed box header (§4.2): its type, the byte offset where its *payload* begins
/// (past the size/type/largesize/uuid), and the byte offset just past its data (its end).
/// All offsets are into the same slice the header was read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoxHeader {
    /// The four-character box type (raw octets — compare against [`boxtype`] constants).
    pub kind: FourCc,
    /// Byte offset where the box payload starts (past the header, largesize, usertype).
    pub body_start: usize,
    /// Byte offset just past this box (== the start of the next sibling).
    pub end: usize,
}

impl BoxHeader {
    /// The box payload as a slice of the buffer it was parsed from. Bounds are guaranteed
    /// valid by [`read_box_header`] (which rejects `end > data.len()`), so this cannot
    /// panic for a header that parse produced from `data`.
    pub fn body<'a>(&self, data: &'a [u8]) -> &'a [u8] {
        &data[self.body_start..self.end]
    }
}

/// Read one box header at `data[at]` (§4.2). Handles the 64-bit `largesize` (size == 1)
/// and skips the 16-byte `usertype` of a `uuid` box. Bounds-checked: a header too short
/// for its declared width, or a box that runs past `data`, is an error (never a panic).
///
/// A `size == 0` box means "to the end of the enclosing container" (§4.2) — its end is
/// `data.len()`. That is only valid for a top-level box (typically the last `mdat`); a
/// caller parsing a child container passes the child window as `data`, so end-of-container
/// resolves correctly there too.
pub fn read_box_header(data: &[u8], at: usize) -> Result<BoxHeader, BoxError> {
    let mut r = Cursor::at(data, at);
    let size32 = r.u32("box size")? as u64;
    let kind_bytes = r.take(4, "box type")?;
    let kind: FourCc = [kind_bytes[0], kind_bytes[1], kind_bytes[2], kind_bytes[3]];

    // Resolve the total box size (§4.2): 1 = 64-bit largesize, 0 = to end of container.
    let total = match size32 {
        1 => r.u64("box largesize")?,
        0 => (data.len() - at) as u64,
        n => n,
    };
    // A `uuid` box carries a 16-octet usertype before its payload (§4.2) — skip it.
    if kind == fourcc(b"uuid") {
        r.skip(16, "uuid usertype")?;
    }
    let body_start = r.at;
    let header_len = (body_start - at) as u64;
    if total < header_len {
        return Err(BoxError::Malformed("box size smaller than its own header"));
    }
    let end = (at as u64)
        .checked_add(total)
        .ok_or(BoxError::Malformed("box end overflows the address space"))?;
    if end > data.len() as u64 {
        return Err(BoxError::Truncated("box runs past the buffer"));
    }
    Ok(BoxHeader { kind, body_start, end: end as usize })
}

/// Iterate the direct child boxes of a container payload, in order, calling `f` for each.
/// Stops on the first [`BoxError`] (propagated). Used to walk `moov`/`trak`/`stbl`/… — a
/// child whose size runs past the container is caught by [`read_box_header`]'s bounds
/// check (its `end > data.len()`).
///
/// A degenerate zero-length trailing region (fewer than 8 bytes left, not enough for a
/// header) ends the walk cleanly rather than erroring — some muxers pad a container.
pub fn for_each_child<F: FnMut(&BoxHeader) -> Result<(), BoxError>>(
    data: &[u8],
    mut f: F,
) -> Result<(), BoxError> {
    let mut at = 0usize;
    while at + 8 <= data.len() {
        let h = read_box_header(data, at)?;
        // A box cannot advance by zero (its size >= its header, checked above), so `at`
        // strictly increases — no infinite loop on adversarial input.
        debug_assert!(h.end > at, "box header must advance the cursor");
        f(&h)?;
        at = h.end;
    }
    Ok(())
}

/// Find the first direct child of `data` whose type is `kind` (§4.2 ordering is not
/// guaranteed, so a linear search). Returns its header, or `None` if absent.
pub fn find_child(data: &[u8], kind: FourCc) -> Result<Option<BoxHeader>, BoxError> {
    let mut found = None;
    for_each_child(data, |h| {
        if found.is_none() && h.kind == kind {
            found = Some(*h);
        }
        Ok(())
    })?;
    Ok(found)
}

// =====================================================================================
// FullBox leaf parsers (§8.x) — each cites its 14496-12 section. Every one is fallible
// and bounds-checked (untrusted input). They take the box *payload* (past the box
// header) and return the fields the sample-table resolver needs.
// =====================================================================================

/// A FullBox version + 24-bit flags prefix (§4.2), consumed by every table box below. The
/// `flags` field is read (so the payload cursor advances past it) but none of the boxes this
/// reader parses branches on a flag value — kept for grammar completeness and a future box
/// (e.g. `tfhd`) that does.
struct FullBoxHead {
    version: u8,
    #[allow(dead_code)]
    flags: u32,
}

/// Read the FullBox `version` + 24-bit `flags` at the start of a full-box payload (§4.2).
fn read_full_box_head(r: &mut Cursor) -> Result<FullBoxHead, BoxError> {
    let version = r.u8("fullbox version")?;
    let hi = r.u8("fullbox flags")? as u32;
    let mid = r.u8("fullbox flags")? as u32;
    let lo = r.u8("fullbox flags")? as u32;
    Ok(FullBoxHead { version, flags: (hi << 16) | (mid << 8) | lo })
}

/// `mvhd` MovieHeaderBox (§8.2.2): the movie-level timescale + duration. Version 0 uses
/// 32-bit times; version 1 uses 64-bit. We read only timescale/duration (the transform
/// matrix and rate/volume are not needed to enumerate samples).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mvhd {
    /// Movie timescale (ticks per second) — the unit `mvhd.duration` and `elst` media
    /// times are expressed in (NOT the media/track timescale in `mdhd`).
    pub timescale: u32,
    /// Movie duration in `timescale` ticks (0 if the file declared none / all-ones v0).
    pub duration: u64,
}

/// Parse an `mvhd` payload (§8.2.2).
pub fn parse_mvhd(body: &[u8]) -> Result<Mvhd, BoxError> {
    let mut r = Cursor::new(body);
    let head = read_full_box_head(&mut r)?;
    match head.version {
        0 => {
            r.skip(8, "mvhd v0 creation/modification time")?;
            let timescale = r.u32("mvhd timescale")?;
            let duration = r.u32("mvhd duration")? as u64;
            Ok(Mvhd { timescale, duration })
        }
        1 => {
            r.skip(16, "mvhd v1 creation/modification time")?;
            let timescale = r.u32("mvhd timescale")?;
            let duration = r.u64("mvhd duration")?;
            Ok(Mvhd { timescale, duration })
        }
        _ => Err(BoxError::Malformed("unsupported mvhd version")),
    }
}

/// `tkhd` TrackHeaderBox (§8.3.2): the presentation width/height (16.16 fixed point) for a
/// video track, plus the track id. We read the id and dimensions; the transform matrix,
/// layer, volume etc. are skipped.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tkhd {
    /// `track_ID` — the 1-based id used by `elst`/references (not the sample route key;
    /// MP4 routes samples by track, one `stbl` per track).
    pub track_id: u32,
    /// Presentation width in integer pixels (the 16.16 fixed-point `width` rounded down),
    /// 0 for a non-visual track.
    pub width: u32,
    /// Presentation height in integer pixels, 0 for a non-visual track.
    pub height: u32,
}

/// Parse a `tkhd` payload (§8.3.2). The width/height are the last two 16.16 fixed-point
/// fields; everything between the times and them (reserved, layer, matrix) is fixed-width
/// and skipped by offset.
pub fn parse_tkhd(body: &[u8]) -> Result<Tkhd, BoxError> {
    let mut r = Cursor::new(body);
    let head = read_full_box_head(&mut r)?;
    let track_id = match head.version {
        0 => {
            r.skip(8, "tkhd v0 creation/modification time")?;
            let id = r.u32("tkhd v0 track_ID")?;
            r.skip(4, "tkhd v0 reserved")?;
            r.skip(4, "tkhd v0 duration")?;
            id
        }
        1 => {
            r.skip(16, "tkhd v1 creation/modification time")?;
            let id = r.u32("tkhd v1 track_ID")?;
            r.skip(4, "tkhd v1 reserved")?;
            r.skip(8, "tkhd v1 duration")?;
            id
        }
        _ => return Err(BoxError::Malformed("unsupported tkhd version")),
    };
    // reserved[2] u32, layer i16, alternate_group i16, volume i16, reserved u16,
    // matrix i32[9] — 8 + 2+2+2+2 + 36 = 52 octets before width/height (§8.3.2).
    r.skip(52, "tkhd fixed fields before dimensions")?;
    let width = r.u32("tkhd width")? >> 16; // 16.16 fixed-point → integer pixels
    let height = r.u32("tkhd height")? >> 16;
    Ok(Tkhd { track_id, width, height })
}

/// `mdhd` MediaHeaderBox (§8.4.2): the **media (track) timescale** — the unit this track's
/// sample times (`stts`/`ctts` deltas) are counted in. This is the timescale used to
/// convert per-sample times to nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mdhd {
    /// Media timescale (ticks per second) for this track's sample timeline.
    pub timescale: u32,
    /// Media duration in `timescale` ticks (0 if none declared).
    pub duration: u64,
}

/// Parse an `mdhd` payload (§8.4.2).
pub fn parse_mdhd(body: &[u8]) -> Result<Mdhd, BoxError> {
    let mut r = Cursor::new(body);
    let head = read_full_box_head(&mut r)?;
    match head.version {
        0 => {
            r.skip(8, "mdhd v0 creation/modification time")?;
            let timescale = r.u32("mdhd timescale")?;
            let duration = r.u32("mdhd duration")? as u64;
            Ok(Mdhd { timescale, duration })
        }
        1 => {
            r.skip(16, "mdhd v1 creation/modification time")?;
            let timescale = r.u32("mdhd timescale")?;
            let duration = r.u64("mdhd duration")?;
            Ok(Mdhd { timescale, duration })
        }
        _ => Err(BoxError::Malformed("unsupported mdhd version")),
    }
}

/// A single `elst` edit-list entry (§8.6.6): a segment of the media presented for
/// `segment_duration` (movie ticks) starting at `media_time` (media ticks; -1 = empty edit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ElstEntry {
    /// Segment duration in **movie** timescale ticks.
    pub segment_duration: u64,
    /// Start time in **media** timescale ticks; `-1` marks an empty edit (a gap / dwell).
    pub media_time: i64,
}

/// Parse an `elst` payload (§8.6.6) into its entries. Version 0 uses 32-bit fields,
/// version 1 uses 64-bit; the media_rate fields (fixed 16.16) are skipped — a demuxer that
/// only shifts the timeline does not need the rate.
pub fn parse_elst(body: &[u8]) -> Result<Vec<ElstEntry>, BoxError> {
    let mut r = Cursor::new(body);
    let head = read_full_box_head(&mut r)?;
    let count = r.u32("elst entry_count")? as usize;
    let mut entries = Vec::with_capacity(count.min(1024)); // cap the pre-alloc on bad input
    for _ in 0..count {
        let (segment_duration, media_time) = match head.version {
            0 => (r.u32("elst v0 segment_duration")? as u64, r.i32("elst v0 media_time")? as i64),
            1 => (r.u64("elst v1 segment_duration")?, r.i64("elst v1 media_time")?),
            _ => return Err(BoxError::Malformed("unsupported elst version")),
        };
        r.skip(4, "elst media_rate")?; // media_rate_integer i16 + media_rate_fraction i16
        entries.push(ElstEntry { segment_duration, media_time });
    }
    Ok(entries)
}

/// One `stts` run (§8.6.1.2): `count` consecutive samples each of `delta` media ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SttsEntry {
    pub count: u32,
    pub delta: u32,
}

/// Parse an `stts` DecodingTimeToSampleBox payload (§8.6.1.2) into its run-length table.
pub fn parse_stts(body: &[u8]) -> Result<Vec<SttsEntry>, BoxError> {
    let mut r = Cursor::new(body);
    let _head = read_full_box_head(&mut r)?;
    let count = r.u32("stts entry_count")? as usize;
    let mut out = Vec::with_capacity(count.min(1 << 20));
    for _ in 0..count {
        let c = r.u32("stts sample_count")?;
        let d = r.u32("stts sample_delta")?;
        out.push(SttsEntry { count: c, delta: d });
    }
    Ok(out)
}

/// One `ctts` run (§8.6.1.3): `count` consecutive samples each with composition `offset`
/// (`pts = dts + offset`). Version 0 offsets are unsigned u32; **version 1 offsets are
/// signed i32** — the common case for streams with B-frames whose composition time can
/// precede decode time. The offset is stored signed here so both versions share one path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CttsEntry {
    pub count: u32,
    pub offset: i64,
}

/// Parse a `ctts` CompositionOffsetBox payload (§8.6.1.3). Version 0: `sample_offset` is
/// `unsigned int(32)`; version 1: `signed int(32)`. We widen to i64 so downstream
/// `pts = dts + offset` arithmetic is uniform and cannot overflow at i32 boundaries.
pub fn parse_ctts(body: &[u8]) -> Result<Vec<CttsEntry>, BoxError> {
    let mut r = Cursor::new(body);
    let head = read_full_box_head(&mut r)?;
    let count = r.u32("ctts entry_count")? as usize;
    let mut out = Vec::with_capacity(count.min(1 << 20));
    for _ in 0..count {
        let c = r.u32("ctts sample_count")?;
        let offset = match head.version {
            0 => r.u32("ctts v0 sample_offset")? as i64,
            1 => r.i32("ctts v1 sample_offset")? as i64,
            _ => return Err(BoxError::Malformed("unsupported ctts version")),
        };
        out.push(CttsEntry { count: c, offset });
    }
    Ok(out)
}

/// The per-sample sizes from `stsz`/`stz2` (§8.7.3): either a single constant size for all
/// samples, or an explicit per-sample table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleSizes {
    /// Every sample is `sample_size` bytes; there are `sample_count` of them.
    Constant { sample_size: u32, sample_count: u32 },
    /// Explicit per-sample sizes (one entry per sample), in sample order.
    Table(Vec<u32>),
}

impl SampleSizes {
    /// The number of samples this table describes.
    pub fn count(&self) -> usize {
        match self {
            SampleSizes::Constant { sample_count, .. } => *sample_count as usize,
            SampleSizes::Table(v) => v.len(),
        }
    }

    /// The size (bytes) of sample `i` (0-based). `None` if out of range.
    pub fn get(&self, i: usize) -> Option<u32> {
        match self {
            SampleSizes::Constant { sample_size, sample_count } => {
                (i < *sample_count as usize).then_some(*sample_size)
            }
            SampleSizes::Table(v) => v.get(i).copied(),
        }
    }
}

/// Parse an `stsz` SampleSizeBox payload (§8.7.3.2): a `sample_size` (0 = per-sample table
/// follows) + `sample_count`, then, if `sample_size == 0`, that many u32 sizes.
pub fn parse_stsz(body: &[u8]) -> Result<SampleSizes, BoxError> {
    let mut r = Cursor::new(body);
    let _head = read_full_box_head(&mut r)?;
    let sample_size = r.u32("stsz sample_size")?;
    let sample_count = r.u32("stsz sample_count")?;
    if sample_size != 0 {
        return Ok(SampleSizes::Constant { sample_size, sample_count });
    }
    let n = sample_count as usize;
    let mut sizes = Vec::with_capacity(n.min(1 << 22));
    for _ in 0..n {
        sizes.push(r.u32("stsz entry_size")?);
    }
    Ok(SampleSizes::Table(sizes))
}

/// Parse an `stz2` CompactSampleSizeBox payload (§8.7.3.3): a `field_size` (4, 8, or 16
/// bits) then `sample_count` packed sizes. A 4-bit field packs two sizes per octet (the
/// first in the high nibble, §8.7.3.3).
pub fn parse_stz2(body: &[u8]) -> Result<SampleSizes, BoxError> {
    let mut r = Cursor::new(body);
    let _head = read_full_box_head(&mut r)?;
    r.skip(3, "stz2 reserved")?; // reserved u24
    let field_size = r.u8("stz2 field_size")?;
    let sample_count = r.u32("stz2 sample_count")? as usize;
    let mut sizes = Vec::with_capacity(sample_count.min(1 << 22));
    match field_size {
        16 => {
            for _ in 0..sample_count {
                sizes.push(r.u16("stz2 16-bit entry")? as u32);
            }
        }
        8 => {
            for _ in 0..sample_count {
                sizes.push(r.u8("stz2 8-bit entry")? as u32);
            }
        }
        4 => {
            // Two 4-bit sizes per octet: high nibble is the earlier sample (§8.7.3.3).
            let mut i = 0;
            while i < sample_count {
                let byte = r.u8("stz2 4-bit pair")?;
                sizes.push((byte >> 4) as u32);
                i += 1;
                if i < sample_count {
                    sizes.push((byte & 0x0F) as u32);
                    i += 1;
                }
            }
        }
        _ => return Err(BoxError::Malformed("stz2 field_size must be 4, 8, or 16")),
    }
    Ok(SampleSizes::Table(sizes))
}

/// One `stsc` SampleToChunkBox run (§8.7.4): starting at chunk `first_chunk` (1-based),
/// each chunk holds `samples_per_chunk` samples described by sample entry
/// `sample_description_index`, until the next run's `first_chunk`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StscEntry {
    pub first_chunk: u32,
    pub samples_per_chunk: u32,
    pub sample_description_index: u32,
}

/// Parse an `stsc` payload (§8.7.4) into its run table.
pub fn parse_stsc(body: &[u8]) -> Result<Vec<StscEntry>, BoxError> {
    let mut r = Cursor::new(body);
    let _head = read_full_box_head(&mut r)?;
    let count = r.u32("stsc entry_count")? as usize;
    let mut out = Vec::with_capacity(count.min(1 << 20));
    for _ in 0..count {
        out.push(StscEntry {
            first_chunk: r.u32("stsc first_chunk")?,
            samples_per_chunk: r.u32("stsc samples_per_chunk")?,
            sample_description_index: r.u32("stsc sample_description_index")?,
        });
    }
    Ok(out)
}

/// Parse an `stco` (32-bit, §8.7.5) or `co64` (64-bit) ChunkOffsetBox payload into the
/// absolute file offset of each chunk, in chunk order. `is_64` selects the field width.
pub fn parse_chunk_offsets(body: &[u8], is_64: bool) -> Result<Vec<u64>, BoxError> {
    let mut r = Cursor::new(body);
    let _head = read_full_box_head(&mut r)?;
    let count = r.u32("chunk-offset entry_count")? as usize;
    let mut out = Vec::with_capacity(count.min(1 << 22));
    for _ in 0..count {
        out.push(if is_64 { r.u64("co64 offset")? } else { r.u32("stco offset")? as u64 });
    }
    Ok(out)
}

/// Parse an `stss` SyncSampleBox payload (§8.6.2) into the 1-based sample numbers that are
/// sync (random-access) points. If `stss` is absent, *every* sample is a sync sample
/// (§8.6.2) — the resolver handles that outside this parser.
pub fn parse_stss(body: &[u8]) -> Result<Vec<u32>, BoxError> {
    let mut r = Cursor::new(body);
    let _head = read_full_box_head(&mut r)?;
    let count = r.u32("stss entry_count")? as usize;
    let mut out = Vec::with_capacity(count.min(1 << 20));
    for _ in 0..count {
        out.push(r.u32("stss sample_number")?);
    }
    Ok(out)
}

/// A minimal bounds-checked big-endian forward cursor over a box slice — every read is
/// fallible so a truncated box errors rather than panicking (spec: untrusted input). MP4
/// scalars are big-endian (§4.2), like EBML and unlike Ogg.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    /// A cursor over the whole slice, positioned at 0.
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    /// A cursor over `data` positioned at `at` (used to read a box header mid-buffer).
    fn at(data: &'a [u8], at: usize) -> Self {
        Self { data, at }
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], BoxError> {
        let end = self.at.checked_add(n).ok_or(BoxError::Malformed(what))?;
        let s = self.data.get(self.at..end).ok_or(BoxError::Truncated(what))?;
        self.at = end;
        Ok(s)
    }

    fn skip(&mut self, n: usize, what: &'static str) -> Result<(), BoxError> {
        let _ = self.take(n, what)?;
        Ok(())
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, BoxError> {
        Ok(self.take(1, what)?[0])
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, BoxError> {
        let b = self.take(2, what)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, BoxError> {
        let b = self.take(4, what)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self, what: &'static str) -> Result<i32, BoxError> {
        Ok(self.u32(what)? as i32)
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, BoxError> {
        let b = self.take(8, what)?;
        Ok(u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn i64(&mut self, what: &'static str) -> Result<i64, BoxError> {
        Ok(self.u64(what)? as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble a box: 8-byte header (size,type) + body. `size` is total incl. header.
    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let total = 8 + body.len();
        let mut out = (total as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn box_header_basic_and_largesize() {
        // A plain 8-byte-header box.
        let b = boxed(b"moov", &[1, 2, 3, 4]);
        let h = read_box_header(&b, 0).unwrap();
        assert_eq!(h.kind, boxtype::MOOV);
        assert_eq!(h.body_start, 8);
        assert_eq!(h.end, b.len());
        assert_eq!(h.body(&b), &[1, 2, 3, 4]);

        // A 64-bit largesize box (size32 == 1, then a u64 total).
        let mut lb = 1u32.to_be_bytes().to_vec();
        lb.extend_from_slice(b"mdat");
        lb.extend_from_slice(&(16u64 + 2).to_be_bytes()); // total = 16 header + 2 body
        lb.extend_from_slice(&[0xAA, 0xBB]);
        let h = read_box_header(&lb, 0).unwrap();
        assert_eq!(h.kind, boxtype::MDAT);
        assert_eq!(h.body_start, 16);
        assert_eq!(h.body(&lb), &[0xAA, 0xBB]);
    }

    #[test]
    fn box_size_zero_runs_to_end() {
        let mut b = 0u32.to_be_bytes().to_vec(); // size 0 → to end of buffer
        b.extend_from_slice(b"mdat");
        b.extend_from_slice(&[9, 9, 9]);
        let h = read_box_header(&b, 0).unwrap();
        assert_eq!(h.end, b.len());
        assert_eq!(h.body(&b), &[9, 9, 9]);
    }

    #[test]
    fn child_walk_visits_each_box_in_order() {
        let mut container = Vec::new();
        container.extend_from_slice(&boxed(b"mvhd", &[0; 4]));
        container.extend_from_slice(&boxed(b"trak", &[0; 8]));
        container.extend_from_slice(&boxed(b"udta", &[0; 2]));
        let mut kinds = Vec::new();
        for_each_child(&container, |h| {
            kinds.push(h.kind);
            Ok(())
        })
        .unwrap();
        assert_eq!(kinds, vec![boxtype::MVHD, boxtype::TRAK, fourcc(b"udta")]);
        assert_eq!(find_child(&container, boxtype::TRAK).unwrap().unwrap().kind, boxtype::TRAK);
        assert!(find_child(&container, boxtype::MOOV).unwrap().is_none());
    }

    #[test]
    fn stts_ctts_stsz_stsc_roundtrip() {
        // stts: two runs.
        let mut stts = vec![0u8; 4]; // version+flags
        stts.extend_from_slice(&2u32.to_be_bytes());
        stts.extend_from_slice(&3u32.to_be_bytes());
        stts.extend_from_slice(&100u32.to_be_bytes());
        stts.extend_from_slice(&1u32.to_be_bytes());
        stts.extend_from_slice(&50u32.to_be_bytes());
        let e = parse_stts(&stts).unwrap();
        assert_eq!(e, vec![SttsEntry { count: 3, delta: 100 }, SttsEntry { count: 1, delta: 50 }]);

        // ctts v1: a negative offset (signed).
        let mut ctts = vec![1u8, 0, 0, 0]; // version 1
        ctts.extend_from_slice(&1u32.to_be_bytes());
        ctts.extend_from_slice(&2u32.to_be_bytes());
        ctts.extend_from_slice(&(-40i32).to_be_bytes());
        let c = parse_ctts(&ctts).unwrap();
        assert_eq!(c, vec![CttsEntry { count: 2, offset: -40 }]);

        // stsz constant.
        let mut stsz = vec![0u8; 4];
        stsz.extend_from_slice(&7u32.to_be_bytes()); // sample_size
        stsz.extend_from_slice(&5u32.to_be_bytes()); // sample_count
        let s = parse_stsz(&stsz).unwrap();
        assert_eq!(s, SampleSizes::Constant { sample_size: 7, sample_count: 5 });
        assert_eq!(s.count(), 5);
        assert_eq!(s.get(4), Some(7));
        assert_eq!(s.get(5), None);

        // stsc one run.
        let mut stsc = vec![0u8; 4];
        stsc.extend_from_slice(&1u32.to_be_bytes());
        stsc.extend_from_slice(&1u32.to_be_bytes()); // first_chunk
        stsc.extend_from_slice(&4u32.to_be_bytes()); // samples_per_chunk
        stsc.extend_from_slice(&1u32.to_be_bytes()); // sample_description_index
        let sc = parse_stsc(&stsc).unwrap();
        assert_eq!(sc, vec![StscEntry { first_chunk: 1, samples_per_chunk: 4, sample_description_index: 1 }]);
    }

    #[test]
    fn stz2_packs_field_sizes() {
        // 4-bit: three samples 0x1, 0x2, 0x3 → bytes 0x12, 0x30.
        let mut b = vec![0u8; 4];
        b.extend_from_slice(&[0, 0, 0]); // reserved u24
        b.push(4); // field_size
        b.extend_from_slice(&3u32.to_be_bytes());
        b.extend_from_slice(&[0x12, 0x30]);
        let s = parse_stz2(&b).unwrap();
        assert_eq!(s, SampleSizes::Table(vec![1, 2, 3]));
    }

    #[test]
    fn malformed_boxes_error_not_panic() {
        // Truncated header.
        assert!(read_box_header(&[0, 0, 0], 0).is_err());
        // Size smaller than header.
        let mut b = 4u32.to_be_bytes().to_vec();
        b.extend_from_slice(b"moov");
        assert!(read_box_header(&b, 0).is_err(), "size 4 < 8-byte header");
        // A child claiming to run past the container.
        let mut c = 100u32.to_be_bytes().to_vec();
        c.extend_from_slice(b"trak");
        assert!(for_each_child(&c, |_| Ok(())).is_err());
        // stsz claiming a huge table with no data → error, not a 4 GiB alloc.
        let mut stsz = vec![0u8; 4];
        stsz.extend_from_slice(&0u32.to_be_bytes()); // per-sample table
        stsz.extend_from_slice(&u32::MAX.to_be_bytes()); // sample_count
        assert!(parse_stsz(&stsz).is_err());
    }
}
