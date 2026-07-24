//! `Mp4Reader` — the demuxer's parse engine: an ISO-BMFF `moov` in (parsed once, up front),
//! then the full file streamed in on the sink pad and sliced into per-track samples by
//! absolute byte offset (spec: `spec/NOTES.md`; ISO/IEC 14496-12). This is the read
//! counterpart the element drives, split into two phases the way MP4 itself is:
//!
//! ## Two phases, because MP4 is not a naturally-streaming container
//! Unlike Matroska (where each Cluster/Block is self-describing and can be decoded as it
//! arrives), an MP4 keeps its **sample tables in `moov`** (`stbl`: `stts`/`ctts`/`stsz`/
//! `stsc`/`stco`/`stss`) and its **sample *bytes* in `mdat`**, addressed by absolute file
//! offset. Nothing in `mdat` can be located until `moov` is fully parsed. So:
//! 1. **Resolution** ([`Mp4Reader::new`]): parse the constructor-supplied file head (which
//!    covers `ftyp` + `moov`) into one [`SampleTable`] per track. Every sample's
//!    `(offset, size, dts, pts, sync)` is materialised here, in **file-offset order**
//!    across all tracks (an interleaved playback order), so streaming is a forward walk.
//! 2. **Streaming** ([`Mp4Reader::push`]/[`next_sample`]): the whole file (from byte 0)
//!    arrives on the sink pad in arbitrary chunks; the reader buffers a bounded window and
//!    emits each resolved sample as soon as its `[offset, offset+size)` bytes are present,
//!    dropping consumed bytes so steady-state memory is one interleave-gap, not the file.
//!
//! ## Untrusted input (spec: "a crash on bad input is a P0")
//! Resolution is fully bounds-checked (via [`crate::boxes`]); a truncated/over-long/
//! self-referential table yields an [`Mp4Error`] the element maps to a `streamcraft`
//! error. Streaming never reads past the buffered window. Fragmented (`moof`) files are
//! detected in resolution and errored loudly — their samples are not in `stbl`.

use crate::boxes::{self, BoxError, BoxHeader};
use crate::codec::{self, Reframer, SampleEntry};

/// A resolution- or streaming-time failure. `Box`/table errors are wrapped from
/// [`BoxError`]; the MP4-specific structural rejections (fragmented file, self-referential
/// chunk map) get their own variants so the element can word the `streamcraft` error.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Mp4Error {
    /// A box or table field was truncated / structurally impossible (from [`crate::boxes`]).
    Box(BoxError),
    /// A required box was absent (`moov`, a track's `stbl`, `stsd`, …).
    Missing(&'static str),
    /// A fragmented-movie file (`mvex`/`moof`, §8.8): samples live in fragments, not `stbl`
    /// — out of v1 scope, errored loudly rather than silently mis-resolved.
    Fragmented,
    /// A sample-table cross-reference is impossible (an `stsc` chunk index past `stco`, a
    /// sample count mismatch that would slice out of range).
    Inconsistent(&'static str),
}

impl From<BoxError> for Mp4Error {
    fn from(e: BoxError) -> Self {
        Mp4Error::Box(e)
    }
}

/// One fully-resolved sample: where its bytes live in the file, its timing, and whether it
/// is a random-access point. Built during resolution; the streaming phase slices `[offset,
/// offset+size)` out of the file for each and hands it to the element with `pts`/`dts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    /// The track this sample belongs to (its 0-based index in [`Mp4Reader::tracks`]).
    pub track_index: usize,
    /// Absolute byte offset of the sample's first byte in the file (`mdat` payload).
    pub offset: u64,
    /// Sample size in bytes.
    pub size: u32,
    /// Decode timestamp in **media-timescale ticks** — the running `stts` sum minus the
    /// edit-list leading media-time shift. **Signed**, because an edit list that trims a
    /// track's composition lead pushes the first sample's DTS negative (the standard
    /// ffmpeg / oxideav convention: the leading `media_time` is subtracted from *both* DTS
    /// and CTS so the first *presented* sample lands at pts 0).
    pub dts: i64,
    /// Presentation (composition) timestamp in **media-timescale ticks**: `dts_raw + ctts
    /// offset - edit_shift`. Signed for the same reason; the ns conversion clamps a negative
    /// presentation time to 0 (the sample presents at stream start).
    pub pts: i64,
    /// True if this is a sync (random-access / key) sample (`stss`, or every sample when
    /// `stss` is absent — §8.6.2).
    pub sync: bool,
}

/// A discovered track's demux-relevant metadata (spec: `stsd` §8.5.2, `mdhd` §8.4.2).
/// Carries what a downstream decoder needs to start plus the sample-entry-derived format.
#[derive(Clone, Debug)]
pub struct Track {
    /// 1-based `track_ID` from `tkhd` (§8.3.2). Informational; samples route by
    /// [`Sample::track_index`] (the 0-based position in [`Mp4Reader::tracks`]).
    pub track_id: u32,
    /// Media timescale (ticks/second) from `mdhd` — the unit `dts`/`pts` are counted in.
    pub timescale: u32,
    /// Track duration in nanoseconds, from `mdhd` duration/timescale (§8.4.2) — what a
    /// remuxing consumer declares as the container duration. 0 when `mdhd` omitted it.
    pub duration_ns: u64,
    /// The parsed first sample entry: codec four-CC, announce family, dims/rate/channels,
    /// the Annex B parameter-set head, and the per-sample [`Reframer`].
    pub entry: SampleEntry,
    /// Presentation width in integer pixels (from `tkhd`, falling back to the sample
    /// entry's coded width). 0 for a non-visual track.
    pub width: u32,
    /// Presentation height in integer pixels. 0 for a non-visual track.
    pub height: u32,
}

impl Track {
    /// The announce family for this track's src pad ([`codec::family_for`]).
    pub fn family(&self) -> &'static str {
        self.entry.family
    }

    /// The Annex B parameter-set head to emit before the first frame (empty for non-NAL /
    /// in-band-parameter-set / raw tracks).
    pub fn codec_head(&self) -> &[u8] {
        &self.entry.codec_head
    }

    /// This track's per-sample reframer (NAL length-prefix → Annex B, or passthrough).
    pub fn reframer(&self) -> &Reframer {
        &self.entry.reframer
    }
}

/// The demuxer's parse engine. [`new`](Self::new) resolves the sample tables from the file
/// head; [`push`](Self::push) feeds the streaming file bytes and [`next_sample`](Self::next_sample)
/// drains resolved samples as their bytes arrive. See the module docs for the two-phase model.
pub struct Mp4Reader {
    /// Per-track metadata, in file order (the order `trak` boxes appear in `moov`).
    tracks: Vec<Track>,
    /// Every sample across all tracks, sorted by file `offset` — the interleaved playback
    /// order, so the streaming walk is strictly forward and the buffer window is bounded by
    /// the largest interleave gap, not the file size.
    samples: Vec<Sample>,
    /// Index into `samples` of the next sample to emit during streaming.
    next: usize,
    /// Absolute file offset of `buf[0]` — the streaming window's base. Advances as consumed
    /// bytes are dropped.
    stream_base: u64,
    /// The buffered streaming window: `file[stream_base .. stream_base + buf.len()]`.
    buf: Vec<u8>,
}

impl Mp4Reader {
    /// Resolve a progressive MP4 from its **file head** — the leading bytes that MUST cover
    /// `ftyp` + the whole `moov` (everything up to, but not necessarily including, `mdat`).
    /// Builds one [`SampleTable`] per track and the file-offset-sorted sample list. The full
    /// file (from byte 0) is fed later via [`push`](Self::push) for the byte slicing.
    ///
    /// Errors (never panics) on a missing/truncated `moov`, a fragmented file (`mvex`/`moof`),
    /// or a self-inconsistent sample table.
    pub fn new(head: &[u8]) -> Result<Self, Mp4Error> {
        let moov = find_top_level(head, boxes::boxtype::MOOV)?
            .ok_or(Mp4Error::Missing("moov"))?;
        let moov_body = moov.body(head);

        // Fragmented movies carry `mvex` in `moov`; their samples are in `moof` fragments,
        // not `stbl`. Detect and refuse (§8.8) rather than resolve an empty table.
        if boxes::find_child(moov_body, boxes::boxtype::MVEX)?.is_some() {
            return Err(Mp4Error::Fragmented);
        }

        // Collect the `trak` box headers first (a BoxError-only walk), then resolve each —
        // resolution returns `Mp4Error`, which `for_each_child`'s BoxError closure cannot
        // carry, so it happens outside the closure.
        let mut trak_boxes = Vec::new();
        boxes::for_each_child(moov_body, |child| {
            if child.kind == boxes::boxtype::TRAK {
                trak_boxes.push(*child);
            }
            Ok(())
        })?;

        let mut tracks = Vec::new();
        let mut samples = Vec::new();
        for trak in &trak_boxes {
            let trak_body = trak.body(moov_body);
            if let Some((track, mut track_samples)) = resolve_track(trak_body, tracks.len())? {
                tracks.push(track);
                samples.append(&mut track_samples);
            }
        }

        if tracks.is_empty() {
            return Err(Mp4Error::Missing("no resolvable track in moov"));
        }

        // Sort by file offset → interleaved playback order (a forward streaming walk).
        // A stable sort keeps same-offset ties (degenerate) in track order.
        samples.sort_by_key(|s| s.offset);

        Ok(Self {
            tracks,
            samples,
            next: 0,
            stream_base: 0,
            buf: Vec::new(),
        })
    }

    /// The discovered tracks, in file order. Complete after [`new`](Self::new).
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// All resolved samples, file-offset-sorted (for tests / oracle cross-validation).
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// Reset the streaming cursor to the file start — call before feeding bytes on `start`.
    /// The resolved tables are retained (resolution happens once, at construction).
    pub fn reset_stream(&mut self) {
        self.next = 0;
        self.stream_base = 0;
        self.buf.clear();
    }

    /// Feed the next chunk of the **whole file** (byte-0-anchored) into the streaming window.
    /// The caller pushes strictly forward, contiguous file bytes; the reader appends them and
    /// drops the consumed prefix so the window stays bounded.
    pub fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Emit the next resolved sample whose bytes are fully buffered, as `(track_index, pts,
    /// dts, sync, bytes)`. Returns `None` when the next sample's bytes have not all arrived
    /// yet (wait for more [`push`](Self::push)) or when every sample has been emitted.
    ///
    /// Samples are served in file-offset order (the resolution sort), so the buffered window
    /// only needs to span from the current sample's offset to the next — the reader drops
    /// everything before the current sample's offset first.
    pub fn next_sample(&mut self) -> Option<ResolvedSample<'_>> {
        if self.next >= self.samples.len() {
            return None;
        }
        let s = self.samples[self.next];
        let start = s.offset;
        let end = s.offset.checked_add(s.size as u64)?;

        // Drop any buffered bytes before this sample (an interleave gap, or slack from the
        // previous sample). This keeps the window bounded to one sample's span.
        if start > self.stream_base {
            let skip = (start - self.stream_base) as usize;
            if skip >= self.buf.len() {
                // The whole window is before this sample — discard it and wait for more.
                self.stream_base += self.buf.len() as u64;
                self.buf.clear();
                return None;
            }
            self.buf.drain(..skip);
            self.stream_base = start;
        }

        // Are the sample's bytes all present?
        let avail_end = self.stream_base + self.buf.len() as u64;
        if end > avail_end {
            return None; // not all arrived yet
        }
        let lo = (start - self.stream_base) as usize;
        let hi = (end - self.stream_base) as usize;
        self.next += 1;
        Some(ResolvedSample {
            track_index: s.track_index,
            pts: s.pts,
            dts: s.dts,
            sync: s.sync,
            bytes: &self.buf[lo..hi],
        })
    }

    /// Convert a (possibly negative) media-timescale tick count for `track_index` to
    /// **non-negative** nanoseconds (i128-safe: `ticks * 1e9` can exceed u64 for long files
    /// at high timescales). A negative presentation time (from an edit-list trim) clamps to 0
    /// — the sample presents at stream start. A zero timescale (malformed `mdhd`) passes the
    /// tick value through (clamped ≥ 0) rather than dividing by zero.
    pub fn ticks_to_ns(&self, track_index: usize, ticks: i64) -> u64 {
        let ts = self.tracks.get(track_index).map(|t| t.timescale).unwrap_or(0);
        ticks_to_ns(ticks, ts)
    }
}

/// A sample the streaming walk produced: its route, timing, sync flag, and a borrow of its
/// bytes out of the reader's window (valid until the next [`Mp4Reader::next_sample`] /
/// [`push`](Mp4Reader::push)).
pub struct ResolvedSample<'a> {
    pub track_index: usize,
    /// Presentation timestamp in media-timescale ticks (signed; use [`Mp4Reader::ticks_to_ns`]
    /// which clamps to non-negative nanoseconds).
    pub pts: i64,
    /// Decode timestamp in media-timescale ticks (signed — may be negative after an edit trim).
    pub dts: i64,
    pub sync: bool,
    pub bytes: &'a [u8],
}

/// Convert `ticks` (signed) at `timescale` ticks/second to **non-negative** nanoseconds via
/// i128 so the intermediate `ticks * 1_000_000_000` never overflows (spec: i128-safe pts).
/// Timescale 0 → passthrough. A negative result clamps to 0.
fn ticks_to_ns(ticks: i64, timescale: u32) -> u64 {
    if timescale == 0 {
        return ticks.max(0) as u64;
    }
    let ns = (ticks as i128 * 1_000_000_000i128) / timescale as i128;
    ns.clamp(0, u64::MAX as i128) as u64
}

/// Find a top-level box of type `kind` in a file head (a flat walk of the head's boxes).
fn find_top_level(head: &[u8], kind: boxes::FourCc) -> Result<Option<BoxHeader>, BoxError> {
    boxes::find_child(head, kind)
}

/// Resolve one `trak`: parse its `mdia`/`minf`/`stbl` tables and the `tkhd` dims, build the
/// per-sample list, and shift the timeline per a single-entry edit list. Returns `None` for
/// a track we skip (no `stbl`, or an empty sample table — a hint/metadata track).
fn resolve_track(trak_body: &[u8], track_index: usize) -> Result<Option<(Track, Vec<Sample>)>, Mp4Error> {
    // tkhd (dimensions + track_id) — optional; a track without one still resolves.
    let tkhd = match boxes::find_child(trak_body, boxes::boxtype::TKHD)? {
        Some(h) => Some(boxes::parse_tkhd(h.body(trak_body))?),
        None => None,
    };

    // Descend trak → mdia → minf → stbl.
    let mdia = boxes::find_child(trak_body, boxes::boxtype::MDIA)?
        .ok_or(Mp4Error::Missing("mdia"))?;
    let mdia_body = mdia.body(trak_body);

    let mdhd = boxes::find_child(mdia_body, boxes::boxtype::MDHD)?
        .ok_or(Mp4Error::Missing("mdhd"))?;
    let mdhd = boxes::parse_mdhd(mdhd.body(mdia_body))?;

    let minf = boxes::find_child(mdia_body, boxes::boxtype::MINF)?
        .ok_or(Mp4Error::Missing("minf"))?;
    let minf_body = minf.body(mdia_body);

    let stbl = boxes::find_child(minf_body, boxes::boxtype::STBL)?
        .ok_or(Mp4Error::Missing("stbl"))?;
    let stbl_body = stbl.body(minf_body);

    // Sample description (stsd) — the codec entry. Required to know the family/reframer.
    let stsd = boxes::find_child(stbl_body, boxes::boxtype::STSD)?
        .ok_or(Mp4Error::Missing("stsd"))?;
    let entry = codec::parse_stsd(stsd.body(stbl_body))?;

    // The timing/size/chunk tables. A track missing any of the mandatory ones (stts, stsz/
    // stz2, stsc, stco/co64) has no locatable samples — skip it rather than error the file
    // (some files carry a chapter/metadata trak with an empty stbl).
    let stts = boxes::find_child(stbl_body, boxes::boxtype::STTS)?;
    let sizes_box = boxes::find_child(stbl_body, boxes::boxtype::STSZ)?
        .or(boxes::find_child(stbl_body, boxes::boxtype::STZ2)?);
    let stsc = boxes::find_child(stbl_body, boxes::boxtype::STSC)?;
    let chunk_offsets_box = boxes::find_child(stbl_body, boxes::boxtype::STCO)?
        .or(boxes::find_child(stbl_body, boxes::boxtype::CO64)?);

    let (Some(stts), Some(sizes_box), Some(stsc), Some(chunk_offsets_box)) =
        (stts, sizes_box, stsc, chunk_offsets_box)
    else {
        return Ok(None); // not a media track we can enumerate samples for
    };

    let stts = boxes::parse_stts(stts.body(stbl_body))?;
    let sizes = if sizes_box.kind == boxes::boxtype::STZ2 {
        boxes::parse_stz2(sizes_box.body(stbl_body))?
    } else {
        boxes::parse_stsz(sizes_box.body(stbl_body))?
    };
    let stsc = boxes::parse_stsc(stsc.body(stbl_body))?;
    let is_64 = chunk_offsets_box.kind == boxes::boxtype::CO64;
    let chunk_offsets = boxes::parse_chunk_offsets(chunk_offsets_box.body(stbl_body), is_64)?;

    let ctts = match boxes::find_child(stbl_body, boxes::boxtype::CTTS)? {
        Some(h) => boxes::parse_ctts(h.body(stbl_body))?,
        None => Vec::new(),
    };
    let stss = match boxes::find_child(stbl_body, boxes::boxtype::STSS)? {
        Some(h) => Some(boxes::parse_stss(h.body(stbl_body))?),
        None => None, // absent → every sample is sync (§8.6.2)
    };

    // Single-entry edit-list media-time shift (see NOTES.md). Absent / empty / multi-entry
    // → zero shift (documented unsupported); a single non-negative media_time shifts pts.
    let media_time_shift = edit_list_shift(trak_body)?;

    let sample_count = sizes.count();
    let samples = build_samples(
        track_index,
        sample_count,
        &sizes,
        &stts,
        &ctts,
        &stsc,
        &chunk_offsets,
        stss.as_deref(),
        media_time_shift,
        mdhd.timescale,
    )?;

    let (track_id, tk_w, tk_h) = tkhd
        .map(|t| (t.track_id, t.width, t.height))
        .unwrap_or((track_index as u32 + 1, 0, 0));
    // Presentation dims: prefer tkhd (the displayed size); fall back to the sample entry's
    // coded size when tkhd omitted them (0).
    let width = if tk_w != 0 { tk_w } else { entry.width };
    let height = if tk_h != 0 { tk_h } else { entry.height };

    let track = Track {
        track_id,
        timescale: mdhd.timescale,
        duration_ns: ticks_to_ns(mdhd.duration as i64, mdhd.timescale),
        entry,
        width,
        height,
    };
    Ok(Some((track, samples)))
}

/// The edit-list **leading media-time shift**, in this track's media ticks (§8.6.6). Returns
/// the `media_time` of the first edit-list entry that is **not** an empty edit (`media_time
/// != -1`) — the standard trim-the-composition-lead convention (ffmpeg / oxideav's
/// `elst_leading_media_time`): that value is subtracted from both DTS and CTS so the first
/// *presented* sample lands at pts 0. `0` for an absent list or an all-empty list.
///
/// The rest of an edit list — multiple segments, dwell/gap semantics, rate changes — is
/// **documented unsupported** (NOTES.md): only this single leading shift is honoured, which
/// is the common case (a lone non-empty entry, or an empty edit followed by one non-empty).
fn edit_list_shift(trak_body: &[u8]) -> Result<i64, Mp4Error> {
    let Some(edts) = boxes::find_child(trak_body, boxes::boxtype::EDTS)? else {
        return Ok(0);
    };
    let Some(elst) = boxes::find_child(edts.body(trak_body), boxes::boxtype::ELST)? else {
        return Ok(0);
    };
    let entries = boxes::parse_elst(elst.body(edts.body(trak_body)))?;
    Ok(entries
        .iter()
        .find(|e| e.media_time != -1)
        .map(|e| e.media_time)
        .unwrap_or(0))
}

/// Fold the `stbl` tables into a flat per-sample list (spec: §8.6.1.2/§8.6.1.3/§8.7.3/
/// §8.7.4/§8.7.5). For each sample this computes its file offset (the `stsc` chunk mapping
/// plus the `stco` chunk bases plus cumulative sizes within a chunk), decode time (the
/// running `stts` sum), presentation time (`dts + ctts` offset, minus the edit shift), and
/// the sync flag.
#[allow(clippy::too_many_arguments)]
fn build_samples(
    track_index: usize,
    sample_count: usize,
    sizes: &boxes::SampleSizes,
    stts: &[boxes::SttsEntry],
    ctts: &[boxes::CttsEntry],
    stsc: &[boxes::StscEntry],
    chunk_offsets: &[u64],
    stss: Option<&[u32]>,
    media_time_shift: i64,
    _timescale: u32,
) -> Result<Vec<Sample>, Mp4Error> {
    if sample_count == 0 {
        return Ok(Vec::new());
    }
    // Expand stsc into a per-chunk samples_per_chunk map. stsc runs are sorted by
    // first_chunk; run i covers chunks [first_chunk_i, first_chunk_{i+1}).
    let chunk_count = chunk_offsets.len();
    if chunk_count == 0 {
        return Err(Mp4Error::Inconsistent("no chunk offsets"));
    }
    // Per-chunk samples_per_chunk, indexed 0-based by chunk. Built by walking stsc runs.
    let mut samples_per_chunk = vec![0u32; chunk_count];
    if stsc.is_empty() {
        return Err(Mp4Error::Inconsistent("empty stsc"));
    }
    for (i, run) in stsc.iter().enumerate() {
        // first_chunk is 1-based; the run applies until the next run's first_chunk (or the
        // last chunk). A first_chunk of 0 or past chunk_count is malformed.
        if run.first_chunk == 0 {
            return Err(Mp4Error::Inconsistent("stsc first_chunk == 0"));
        }
        let start = (run.first_chunk - 1) as usize;
        if start >= chunk_count {
            return Err(Mp4Error::Inconsistent("stsc first_chunk past chunk count"));
        }
        let end = match stsc.get(i + 1) {
            Some(next) => {
                if next.first_chunk <= run.first_chunk {
                    return Err(Mp4Error::Inconsistent("stsc first_chunk not increasing"));
                }
                ((next.first_chunk - 1) as usize).min(chunk_count)
            }
            None => chunk_count,
        };
        for spc in samples_per_chunk.iter_mut().take(end).skip(start) {
            *spc = run.samples_per_chunk;
        }
    }

    // Sanity: the total samples the chunk map describes must equal sample_count. A mismatch
    // means the tables disagree — reject rather than slice out of range.
    let mapped: u64 = samples_per_chunk.iter().map(|&n| n as u64).sum();
    if mapped != sample_count as u64 {
        return Err(Mp4Error::Inconsistent(
            "stsc/stco sample total != stsz sample count",
        ));
    }

    // stts → per-sample dts (running sum). Iterate the run-length table lazily.
    let mut stts_iter = RunLengthTimes::new(stts);
    // ctts → per-sample composition offset. Absent (empty) → offset 0 for every sample.
    let mut ctts_iter = RunLengthOffsets::new(ctts);

    // stss sync set (1-based sample numbers). Absent → every sample sync.
    let sync_lookup = SyncLookup::new(stss);

    let mut out = Vec::with_capacity(sample_count);
    let mut sample_no = 0usize; // 0-based across the whole track
    // The raw decode time (running `stts` sum) before the edit-list shift, in ticks.
    let mut dts_raw: u64 = 0;

    for (chunk_idx, &chunk_off) in chunk_offsets.iter().enumerate() {
        let spc = samples_per_chunk[chunk_idx];
        let mut offset = chunk_off;
        for _ in 0..spc {
            if sample_no >= sample_count {
                return Err(Mp4Error::Inconsistent("more chunk samples than sizes"));
            }
            let size = sizes
                .get(sample_no)
                .ok_or(Mp4Error::Inconsistent("sample index past size table"))?;
            let delta = stts_iter.next_delta().ok_or(Mp4Error::Inconsistent(
                "stts describes fewer samples than stsz",
            ))?;
            let ctts_off = ctts_iter.next_offset();

            // The edit-list leading media-time is subtracted from BOTH decode and composition
            // time (the ffmpeg / oxideav convention), so the first presented sample lands at
            // pts 0. All i128-safe, then narrowed to i64 (well within range for real files):
            //   dts = dts_raw - shift ; pts = dts_raw + ctts_off - shift.
            let dts_i = dts_raw as i128 - media_time_shift as i128;
            let pts_i = dts_raw as i128 + ctts_off as i128 - media_time_shift as i128;
            let dts = dts_i.clamp(i64::MIN as i128, i64::MAX as i128) as i64;
            let pts = pts_i.clamp(i64::MIN as i128, i64::MAX as i128) as i64;

            let sync = sync_lookup.is_sync(sample_no);

            out.push(Sample {
                track_index,
                offset,
                size,
                dts,
                pts,
                sync,
            });

            dts_raw = dts_raw.checked_add(delta as u64).ok_or(Mp4Error::Inconsistent(
                "cumulative dts overflow",
            ))?;
            offset = offset
                .checked_add(size as u64)
                .ok_or(Mp4Error::Inconsistent("chunk offset overflow"))?;
            sample_no += 1;
        }
    }

    if sample_no != sample_count {
        return Err(Mp4Error::Inconsistent("chunk map produced wrong sample count"));
    }
    Ok(out)
}

/// A lazy run-length walker over `stts` deltas: yields one `delta` per sample in order.
struct RunLengthTimes<'a> {
    runs: &'a [boxes::SttsEntry],
    run: usize,
    left: u32,
}

impl<'a> RunLengthTimes<'a> {
    fn new(runs: &'a [boxes::SttsEntry]) -> Self {
        Self { runs, run: 0, left: runs.first().map(|e| e.count).unwrap_or(0) }
    }

    /// The next sample's delta, or `None` if the table is exhausted.
    fn next_delta(&mut self) -> Option<u32> {
        // Skip zero-count runs (degenerate but legal) until a run with samples or the end.
        while self.left == 0 {
            self.run += 1;
            self.left = self.runs.get(self.run)?.count;
        }
        self.left -= 1;
        Some(self.runs[self.run].delta)
    }
}

/// A lazy run-length walker over `ctts` composition offsets: yields one `offset` per sample.
/// An empty table (no `ctts` box) yields `0` for every sample (pts == dts).
struct RunLengthOffsets<'a> {
    runs: &'a [boxes::CttsEntry],
    run: usize,
    left: u32,
}

impl<'a> RunLengthOffsets<'a> {
    fn new(runs: &'a [boxes::CttsEntry]) -> Self {
        Self { runs, run: 0, left: runs.first().map(|e| e.count).unwrap_or(0) }
    }

    /// The next sample's composition offset (0 once the table is exhausted / absent).
    fn next_offset(&mut self) -> i64 {
        while self.left == 0 {
            self.run += 1;
            match self.runs.get(self.run) {
                Some(e) => self.left = e.count,
                None => return 0, // exhausted → remaining samples have zero offset
            }
        }
        self.left -= 1;
        self.runs[self.run].offset
    }
}

/// Sync-sample lookup over the (sorted, 1-based) `stss` list, or "all sync" when absent.
struct SyncLookup<'a> {
    /// `Some(sorted 1-based sample numbers)` → only those are sync; `None` → every sample.
    list: Option<&'a [u32]>,
}

impl<'a> SyncLookup<'a> {
    fn new(list: Option<&'a [u32]>) -> Self {
        Self { list }
    }

    /// Is the 0-based sample `i` a sync sample? Binary-searches the `stss` list (which is
    /// stored 1-based). `None` list → every sample is a sync sample (§8.6.2).
    fn is_sync(&self, i: usize) -> bool {
        match self.list {
            None => true,
            Some(list) => list.binary_search(&((i + 1) as u32)).is_ok(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boxes::{CttsEntry, SampleSizes, StscEntry, SttsEntry};

    #[test]
    fn run_length_times_walks_deltas() {
        let stts = [SttsEntry { count: 2, delta: 100 }, SttsEntry { count: 1, delta: 40 }];
        let mut it = RunLengthTimes::new(&stts);
        assert_eq!(it.next_delta(), Some(100));
        assert_eq!(it.next_delta(), Some(100));
        assert_eq!(it.next_delta(), Some(40));
        assert_eq!(it.next_delta(), None);
    }

    #[test]
    fn run_length_offsets_exhaust_to_zero() {
        let ctts = [CttsEntry { count: 1, offset: 200 }, CttsEntry { count: 1, offset: -50 }];
        let mut it = RunLengthOffsets::new(&ctts);
        assert_eq!(it.next_offset(), 200);
        assert_eq!(it.next_offset(), -50);
        assert_eq!(it.next_offset(), 0, "past the table → zero");
        // Absent ctts → all zeros.
        let mut it = RunLengthOffsets::new(&[]);
        assert_eq!(it.next_offset(), 0);
    }

    #[test]
    fn sync_lookup_absent_is_all_sync() {
        let s = SyncLookup::new(None);
        assert!(s.is_sync(0) && s.is_sync(7));
        let list = [1u32, 4];
        let s = SyncLookup::new(Some(&list));
        assert!(s.is_sync(0), "sample 1 (0-based 0) is sync");
        assert!(!s.is_sync(1));
        assert!(s.is_sync(3), "sample 4 (0-based 3) is sync");
    }

    #[test]
    fn ticks_to_ns_is_i128_safe() {
        // A large tick count at a high timescale that would overflow u64 * 1e9.
        assert_eq!(ticks_to_ns(0, 48_000), 0);
        assert_eq!(ticks_to_ns(48_000, 48_000), 1_000_000_000);
        // 90 kHz clock, ~2^40 ticks — the intermediate exceeds u64 but i128 holds it.
        let big = 1i64 << 40;
        let want = (big as i128 * 1_000_000_000 / 90_000) as u64;
        assert_eq!(ticks_to_ns(big, 90_000), want);
        assert_eq!(ticks_to_ns(1234, 0), 1234, "timescale 0 → passthrough");
        // A negative tick (edit-list trim) clamps to 0 nanoseconds.
        assert_eq!(ticks_to_ns(-9000, 90_000), 0, "negative presentation time → 0");
        assert_eq!(ticks_to_ns(-5, 0), 0, "timescale 0 + negative → 0");
    }

    /// Build a two-chunk, single-run sample table and check offsets/dts/pts fold correctly.
    #[test]
    fn build_samples_folds_offsets_and_times() {
        // 4 samples, two chunks of 2; sizes 10,20,30,40; chunk offsets 1000, 5000.
        let sizes = SampleSizes::Table(vec![10, 20, 30, 40]);
        let stts = [SttsEntry { count: 4, delta: 512 }];
        let ctts = [CttsEntry { count: 4, offset: 0 }];
        let stsc = [StscEntry { first_chunk: 1, samples_per_chunk: 2, sample_description_index: 1 }];
        let chunk_offsets = [1000u64, 5000];
        let out = build_samples(0, 4, &sizes, &stts, &ctts, &stsc, &chunk_offsets, None, 0, 512).unwrap();
        assert_eq!(out.len(), 4);
        // Chunk 0 at 1000: sample 0 @1000 (10), sample 1 @1010 (20).
        assert_eq!((out[0].offset, out[0].size, out[0].dts, out[0].pts), (1000, 10, 0, 0));
        assert_eq!((out[1].offset, out[1].size, out[1].dts, out[1].pts), (1010, 20, 512, 512));
        // Chunk 1 at 5000: sample 2 @5000 (30), sample 3 @5030 (40).
        assert_eq!((out[2].offset, out[2].size, out[2].dts), (5000, 30, 1024));
        assert_eq!((out[3].offset, out[3].size, out[3].dts), (5030, 40, 1536));
        // stss None → all sync.
        assert!(out.iter().all(|s| s.sync));
    }

    /// A ctts shifts pts relative to dts; the edit-list leading media-time is subtracted from
    /// BOTH dts and pts (the ffmpeg / oxideav convention), so the first presented sample lands
    /// at pts 0 while its dts goes negative. Signed throughout.
    #[test]
    fn build_samples_applies_ctts_and_edit_shift() {
        let sizes = SampleSizes::Constant { sample_size: 5, sample_count: 3 };
        let stts = [SttsEntry { count: 3, delta: 100 }];
        // ctts: first sample +200, then two at 0 (typical B-frame lead).
        let ctts = [CttsEntry { count: 1, offset: 200 }, CttsEntry { count: 2, offset: 0 }];
        let stsc = [StscEntry { first_chunk: 1, samples_per_chunk: 3, sample_description_index: 1 }];
        let chunk_offsets = [100u64];
        // Edit shift of 200 media ticks (trim the composition lead so presentation starts at 0).
        let out = build_samples(0, 3, &sizes, &stts, &ctts, &stsc, &chunk_offsets, None, 200, 1).unwrap();
        // sample 0: dts_raw 0 → dts -200; pts 0+200-200 = 0.
        assert_eq!((out[0].dts, out[0].pts), (-200, 0));
        // sample 1: dts_raw 100 → dts -100; pts 100+0-200 = -100.
        assert_eq!((out[1].dts, out[1].pts), (-100, -100));
        // sample 2: dts_raw 200 → dts 0; pts 200+0-200 = 0.
        assert_eq!((out[2].dts, out[2].pts), (0, 0));
    }

    /// A chunk map whose sample total disagrees with stsz is rejected (not sliced out of range).
    #[test]
    fn build_samples_rejects_inconsistent_tables() {
        let sizes = SampleSizes::Constant { sample_size: 5, sample_count: 4 };
        let stts = [SttsEntry { count: 4, delta: 100 }];
        // stsc says 2 samples/chunk over 1 chunk = 2, but stsz says 4 → mismatch.
        let stsc = [StscEntry { first_chunk: 1, samples_per_chunk: 2, sample_description_index: 1 }];
        let chunk_offsets = [100u64];
        assert!(build_samples(0, 4, &sizes, &stts, &[], &stsc, &chunk_offsets, None, 0, 1).is_err());
    }
}
