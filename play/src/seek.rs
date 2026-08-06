//! Building the pipeline's time→byte [`SeekIndex`] (spec: flush/seek — mapping a time to a
//! byte offset is the seek issuer's job; a byte source has no notion of time, so the
//! controller supplies the map). Also surfaces the declared duration for the CLI's
//! digit-seek and progress.
//!
//! Per container:
//! - **MKV**: the front `SeekHead` (in the header prefix) points at the `Cues`; pread that
//!   range and parse it into `(time_ns, byte)` keyframe cluster entries. A cue-less file
//!   gets an empty index — [`SeekIndex::resolve`] then falls back to the proportional
//!   `file_len`/`duration` estimate, safe because the demuxer re-syncs by cluster scan.
//!   Lifted from `sdl3/examples/play_file.rs`.
//! - **MP4**: the `stss` sync-sample table falls straight out of `Mp4Reader` — each sync
//!   sample carries an absolute file `offset`, a `pts` in media ticks, and a `sync` flag
//!   (ISO/IEC 14496-12 §8.6.2). One entry per video keyframe gives an *exact* seek index,
//!   strictly better than the proportional fallback.
//! - **MP3**: the Xing/Info **seek TOC** in the first frame — 100 points mapping hundredths
//!   of the duration to 256ths of the audio region, which is as much of an index as a
//!   container-less format has. Without one, a hundred-point linear ramp over the audio
//!   region: the constant-bitrate assumption, anchored past the ID3v2 tag rather than at
//!   byte 0 (which is what the bare proportional fallback would do, and why it is not used
//!   here — a 2 MiB cover image in front of a 3 MiB song would skew every seek by 40 %).
//! - **FLAC**: the `SEEKTABLE` (RFC 9639 §8.5), an *exact* `(sample, byte)` index the
//!   encoder writes by default — the elementary-stream equivalent of MKV's Cues.
//! - **WAV**: PCM is constant-rate, so time↔byte is an exact straight line; the entries are
//!   that line sampled on a ~100 ms grid, each landing on a whole interchannel frame.
//! - **Ogg**: no index; the proportional fallback, now that a tail read gives it a duration
//!   to be proportional *to*. Bisecting the page granules for an exact landing is a
//!   documented follow-up, not v1.
//! - **ADTS AAC**: nothing. An ADTS stream states no duration anywhere and carries no
//!   index; the only way to either is walking every frame header in the file, which is not
//!   a cost worth paying at open time for a format nothing in a music library uses.
//!
//! This is controller code running before the pipeline; the file reads it needs (Cues
//! pread, whole-head reparse, and [`crate::head`]'s elementary head/tail windows) carry the
//! documented `#[allow]` per `clippy.toml`.

use profluens_core::pipeline::SeekIndex;

/// A built index plus the declared stream duration in ns (for the CLI's digit-seek and the
/// summary). `duration` is `None` when the container did not declare one.
pub struct SeekInfo {
    pub index: SeekIndex,
    pub duration_ns: Option<u64>,
}

impl SeekInfo {
    /// An index with only the proportional fallback armed (entries empty): time seeking
    /// works approximately when `file_len` and a duration are known, else not at all. The
    /// honest default for containers/streams without a keyframe index.
    fn proportional(file_len: u64, duration_ns: Option<u64>) -> Self {
        SeekInfo {
            index: SeekIndex { entries: Vec::new(), file_len: Some(file_len) },
            duration_ns,
        }
    }
}

/// Build the MKV time→byte index (spec: flush/seek). `header` is the same prefix
/// `MkvDemux::new` was handed (through the first Cluster); `file_len` is the source size.
///
/// The declared duration comes from the header's Segment `Info` (the same probe the
/// demuxer runs). The `SeekHead` → `Cues` chain gives keyframe-accurate entries; absent
/// Cues leave the entries empty (proportional fallback). Lifted from play_file's
/// `build_seek_index`, with the pread bounded to 4 MiB (the Cues master is small).
pub fn mkv_seek_index(path: &str, header: &[u8], file_len: u64) -> SeekInfo {
    // Duration from the header's Segment Info — a throwaway reader parses the same prefix.
    let mut probe = pf_mkv::MatroskaReader::new();
    let _ = probe.push(header);
    let duration_ns = probe.duration_ns();

    let mut index = SeekIndex { entries: Vec::new(), file_len: Some(file_len) };
    let Some(info) = pf_mkv::parse_seek_head(header) else {
        return SeekInfo { index, duration_ns };
    };
    if let Some(cues_pos) = info.cues_pos {
        let abs = info.segment_data_start + cues_pos;
        if abs < file_len {
            // The Cues master is a few bytes per keyframe cluster; read a bounded chunk
            // from its offset — `parse_cues` reads the element's own declared size and
            // ignores the tail, so an over-read is harmless.
            let want = ((file_len - abs) as usize).min(4 * 1024 * 1024);
            if let Some(buf) = pread(path, abs, want) {
                index.entries = pf_mkv::parse_cues(&buf, info.segment_data_start, info.timestamp_scale);
            }
        }
    }
    SeekInfo { index, duration_ns }
}

/// Build the MP4 time→byte index from the resolved sample tables (spec: flush/seek). One
/// entry per *video* sync sample: its presentation time (ticks → ns via the track's
/// timescale) mapped to its absolute file `offset`. This is keyframe-exact — the seek
/// issuer floors to the preceding keyframe cluster and the demuxer resumes there cleanly.
///
/// Falls back to the proportional index when there is no video track (audio-only MP4: an
/// audio sample is a random-access point anyway, so proportional-by-bytes is adequate and
/// the demuxer re-walks its sample list). The declared duration is the max track duration.
pub fn mp4_seek_index(reader: &pf_mp4::Mp4Reader, file_len: u64) -> SeekInfo {
    // Duration: the longest track (mdhd duration/timescale, already in ns on the Track).
    let duration_ns = reader.tracks().iter().map(|t| t.duration_ns).max().filter(|&d| d > 0);

    // Pick the first video track by dimensions (a visual track has non-zero width/height).
    let video_index = reader
        .tracks()
        .iter()
        .position(|t| t.width != 0 && t.height != 0);
    let Some(vidx) = video_index else {
        return SeekInfo::proportional(file_len, duration_ns);
    };
    let timescale = reader.tracks()[vidx].timescale.max(1) as u64;

    // `samples()` is sorted by file offset (interleaved playback order); we want ascending
    // *time*, so collect the video sync samples and sort by pts. A negative pts (edit-list
    // composition lead) clamps to 0 — the sample presents at stream start.
    let mut entries: Vec<(u64, u64)> = reader
        .samples()
        .iter()
        .filter(|s| s.track_index == vidx && s.sync)
        .map(|s| {
            let pts_ticks = s.pts.max(0) as u64;
            let time_ns = pts_ticks.saturating_mul(1_000_000_000) / timescale;
            (time_ns, s.offset)
        })
        .collect();
    entries.sort_unstable_by_key(|(t, _)| *t);
    // Dedup identical times (defensive — two sync samples should not share a pts).
    entries.dedup_by_key(|(t, _)| *t);

    SeekInfo {
        index: SeekIndex { entries, file_len: Some(file_len) },
        duration_ns,
    }
}

/// Build the AVI seek info (spec: flush/seek). v1 is the **proportional** fallback: the
/// declared duration comes from the AVI header (the video stream's frame count × its
/// `dwScale/dwRate` sample duration, via `pf-avi`'s [`AviHeader::duration_ns`]), and
/// [`SeekIndex::resolve`] estimates a byte offset from `file_len`; the demuxer re-syncs to
/// the next chunk id after a `FlushStart`, so an approximate landing is safe.
///
/// Keyframe-exact AVI seeking (the trailing `idx1` → `pf_avi::build_seek_index`) is a
/// documented follow-up: it needs an app-side pread of the `idx1` range at the file tail
/// (like the MKV Cues pread), orthogonal to getting AVI *playback* working.
///
/// [`AviHeader::duration_ns`]: pf_avi::AviHeader::duration_ns
pub fn avi_seek_index(header: &[u8], file_len: u64) -> SeekInfo {
    let duration_ns = pf_avi::probe_header(header).ok().and_then(|(h, _movi)| h.duration_ns());
    SeekInfo::proportional(file_len, duration_ns)
}

// --- Elementary streams ---------------------------------------------------------------
//
// A container states its duration in a header field and its index in a dedicated element.
// An elementary stream has neither, so each of these digs the same two answers out of
// whatever the *codec* happened to leave lying around — and the shape of that is different
// for every one of them. `head` reads the windows; the parsing is the codec crates'.

/// Build the MP3 seek info from the head and tail windows [`crate::head::mp3_head`] and
/// [`crate::head::read_tail`] read (spec: flush/seek).
///
/// **Duration** comes from `pf_mp3::props::probe_props`: exact when the first frame carries
/// a Xing/Info or VBRI frame count (less the encoder delay and padding a LAME extension
/// states), and the constant-bitrate estimate from the audio region's size otherwise. The
/// audio region is the file *minus its tags* — an ID3v2 tag in front, an APEv2 and/or
/// ID3v1 block behind — because counting a cover image as audio inflates that estimate.
///
/// **The index** is the Xing seek TOC when there is one, else the linear ramp described in
/// the module docs. Both are anchored on the first frame's own file offset, so a tagged
/// file's map is not shifted by the size of its tag.
///
/// `tail` must end at EOF and begin at `tail_base`; a window that does not is treated as
/// having no trailing tags rather than guessing where they would be.
// One-time: an index is built once per file at open, before any pipeline exists — the
// same sanctioned exception the head/tail reads above run under (spec: allocation
// discipline — setup code, not a `process()` hot path).
#[allow(clippy::disallowed_methods)]
pub fn mp3_seek_index(head: &[u8], tail: &[u8], tail_base: u64, file_len: u64) -> SeekInfo {
    let none = SeekInfo::proportional(file_len, None);
    let v2 = pf_mp3::id3::v2_total_len(head).map_or(0, |n| n as u64).min(file_len);
    let trailers = mp3_trailers(tail, tail_base, file_len).min(file_len - v2);
    let audio_len = file_len - v2 - trailers;

    let Some(window) = usize::try_from(v2).ok().and_then(|at| head.get(at..)) else {
        return none; // the head does not even reach past the tag
    };
    let Some(props) = pf_mp3::probe_props(window, audio_len) else {
        return none; // no MPEG frame in the window: nothing to derive anything from
    };
    let Some(duration_ns) = props.duration_ns.filter(|&d| d > 0) else {
        return none; // free format with no VBR header — neither a count nor a bitrate
    };

    // Where the audio really starts, and how many bytes of it there are: the two ends of
    // the line (or the table) every entry below is drawn on.
    let audio_start = v2.saturating_add(props.first_frame_offset as u64).min(file_len);
    let span = (file_len - trailers).saturating_sub(audio_start);
    // The TOC's fractions are of the length the *encoder* measured, which it states in the
    // header whenever it states anything; our own measurement of the audio region is the
    // fallback. They agree unless something was appended after encoding — and if the
    // encoder's figure is larger than the audio actually present, ours is the honest one.
    let stream_bytes =
        props.stream_bytes.map(u64::from).filter(|&b| b > 0 && b <= span).unwrap_or(span);

    let mut entries = Vec::with_capacity(pf_mp3::XingToc::LEN);
    match props.toc {
        Some(toc) => {
            for i in 0..pf_mp3::XingToc::LEN {
                let (Some(time), Some(off)) =
                    (toc.time_at(i, duration_ns), toc.byte_at(i, stream_bytes))
                else {
                    continue;
                };
                push_ascending(&mut entries, time, audio_start.saturating_add(off), file_len);
            }
        }
        None => linear_ramp(&mut entries, duration_ns, audio_start, span, file_len),
    }
    SeekInfo {
        index: SeekIndex { entries, file_len: Some(file_len) },
        duration_ns: Some(duration_ns),
    }
}

/// An ID3v1 tag is a fixed 128-byte block at the very end of the file (Eric Kemp's 1996
/// `id3v1` note).
const ID3V1_LEN: u64 = 128;

/// Bytes at the end of an MP3 that are metadata rather than audio: an APEv2 tag and/or an
/// ID3v1 block, in that order. ID3v1 is by definition the last 128 bytes, and the APEv2
/// specification's "Tag location" puts an APE tag *in front of* it — a ReplayGain tool must
/// be able to write one without disturbing a trailing v1 block — so the APE footer has to be
/// looked for at `file_len - 128` as well as at `file_len`.
///
/// Counting these as audio would both inflate a bitrate-derived duration and stretch the
/// byte map across bytes that never decode. (This is the same measurement `pf-tags`'
/// `tail::measure` makes for the four formats that share the layout; the player needs only
/// the total, and needs it without depending on a tag scanner.)
fn mp3_trailers(tail: &[u8], base: u64, file_len: u64) -> u64 {
    // Every offset below is derived from `file_len`, so a window that does not end at EOF
    // cannot be measured at all — report nothing rather than measure the wrong bytes.
    if file_len.checked_sub(base) != Some(tail.len() as u64) {
        return 0;
    }
    let rel = |abs: u64| -> Option<usize> {
        usize::try_from(abs.checked_sub(base)?).ok().filter(|&d| d <= tail.len())
    };
    let v1 = match file_len.checked_sub(ID3V1_LEN).and_then(&rel) {
        Some(at) if tail[at..].starts_with(b"TAG") => ID3V1_LEN,
        _ => 0,
    };
    let ape_end = file_len - v1;
    // `ape_tail_len` reads the *last* 32 bytes of the slice it is given, so the slice ends
    // exactly where the APE tag would.
    let ape = rel(ape_end)
        .and_then(|at| pf_mp3::id3::ape_tail_len(&tail[..at]))
        .map(|n| n as u64)
        .filter(|&n| n <= ape_end)
        .unwrap_or(0);
    v1 + ape
}

/// Build the FLAC seek info from the metadata chain [`crate::head::flac_head`] read (spec:
/// flush/seek).
///
/// **Duration** is exact: `STREAMINFO` (RFC 9639 §8.2) states the stream's total
/// interchannel sample count outright, in a 36-bit field, and dividing by the sample rate
/// is the whole calculation. A 0 there means "unknown" (§8.2 permits it, and a stream
/// encoded to a pipe has it), and then there is no duration to report.
///
/// **The index** is the `SEEKTABLE` (§8.5) — one entry per seek point, its sample number
/// turned into a time and its offset made file-absolute by adding the end of the metadata
/// chain. That is an exact, encoder-provided time→byte map, and the reference encoder
/// writes one (a point per second) unless told not to. A stream with no SEEKTABLE, or a
/// head window that did not reach the end of the chain, leaves the entries empty and falls
/// back to the proportional estimate — which is decent for FLAC, whose bitrate varies far
/// less than a lossy codec's.
// One-time: an index is built once per file at open, before any pipeline exists — the
// same sanctioned exception the head/tail reads above run under (spec: allocation
// discipline — setup code, not a `process()` hot path).
#[allow(clippy::disallowed_methods)]
pub fn flac_seek_index(head: &[u8], file_len: u64) -> SeekInfo {
    let Some(info) = pf_flac::tags::stream_info(head) else {
        return SeekInfo::proportional(file_len, None);
    };
    let duration_ns = (info.total_samples > 0)
        .then(|| samples_to_ns(info.total_samples, info.sample_rate))
        .flatten();

    let mut entries = Vec::new();
    if let Some(start) = pf_flac::tags::audio_start(head) {
        let start = start as u64;
        for p in pf_flac::tags::seek_table(head) {
            let (Some(time), Some(byte)) =
                (samples_to_ns(p.sample, info.sample_rate), start.checked_add(p.byte_offset))
            else {
                continue;
            };
            push_ascending(&mut entries, time, byte, file_len);
        }
    }
    SeekInfo { index: SeekIndex { entries, file_len: Some(file_len) }, duration_ns }
}

/// Build the WAV seek info from the header prefix (spec: flush/seek).
///
/// Both halves are exact and for the same reason: PCM is constant-rate, so the `fmt ` byte
/// rate and the `data` chunk size (RIFF/WAVE, Microsoft's "Multimedia Programming
/// Interface and Data Specification 1.0", the `fmt `/`data` chunk layout) give the duration
/// by division and any time its byte by multiplication.
///
/// The entries are that straight line sampled on a grid rather than left to
/// [`SeekIndex`]'s proportional fallback, which would be *nearly* right and wrong in two
/// ways that matter: it measures from byte 0, so the header (and any `LIST`/`INFO` chunk
/// in front of `data`) shifts every landing; and it lands on an arbitrary byte, so a stereo
/// s16 file resumes mid-frame and plays with its channels swapped from there on. Each entry
/// here starts at the `data` chunk and is floored to a whole interchannel frame, and its
/// time is the *exact* time of that byte rather than the grid point that asked for it.
///
/// The grid is 100 ms, thinned for a long file so the index cannot exceed
/// [`WAV_MAX_ENTRIES`] points (~13 minutes at full density; an hour-long file gets ~440 ms).
/// [`SeekIndex::resolve`] floors to the preceding entry and reports where it landed, so the
/// coarseness is an honest sub-second rounding, never drift.
// One-time: an index is built once per file at open, before any pipeline exists — the
// same sanctioned exception the head/tail reads above run under (spec: allocation
// discipline — setup code, not a `process()` hot path).
#[allow(clippy::disallowed_methods)]
pub fn wav_seek_index(head: &[u8], file_len: u64) -> SeekInfo {
    let Ok(h) = profluens_audio::parse_wav_header(head) else {
        return SeekInfo::proportional(file_len, None);
    };
    let stride = h.format.frame_stride() as u64;
    let data_start = h.data_offset as u64;
    if stride == 0 || h.format.sample_rate == 0 || data_start >= file_len {
        return SeekInfo::proportional(file_len, None);
    }
    let byte_rate = u64::from(h.format.sample_rate).saturating_mul(stride);
    // The declared payload size, clamped to what the file actually holds: a truncated
    // download still declares the length it was going to be.
    let data_len = h.data_len.unwrap_or(u64::MAX).min(file_len - data_start);
    let Some(duration_ns) = samples_to_ns(data_len / stride, h.format.sample_rate) else {
        return SeekInfo::proportional(file_len, None);
    };

    let step_ns = WAV_GRID_NS.max(duration_ns / WAV_MAX_ENTRIES + 1);
    let mut entries = Vec::new();
    let mut at_ns = 0u64;
    while at_ns < duration_ns {
        let byte = (u128::from(at_ns) * u128::from(byte_rate) / 1_000_000_000) as u64;
        let aligned = byte - byte % stride;
        let Some(time) = samples_to_ns(aligned / stride, h.format.sample_rate) else { break };
        push_ascending(&mut entries, time, data_start + aligned, file_len);
        at_ns = at_ns.saturating_add(step_ns);
    }
    SeekInfo {
        index: SeekIndex { entries, file_len: Some(file_len) },
        duration_ns: Some(duration_ns),
    }
}

/// Nominal spacing of the WAV index's entries — see [`wav_seek_index`].
pub const WAV_GRID_NS: u64 = 100_000_000;
/// Cap on the WAV index's entry count, so an hours-long file's index stays small.
pub const WAV_MAX_ENTRIES: u64 = 8_192;

/// Build the Ogg seek info: no index, but a **duration**, from the identification header on
/// the first (bos) page and the granule position on the last (spec: flush/seek).
///
/// Ogg has no duration field. RFC 3533 §6 defines the granule position as "the total
/// samples encoded after including all packets finished on this page", so the length of a
/// stream *is* the granule of its last page, in whatever unit the codec mapping counts in —
/// which is why this needs both ends of the file and why the answer differs per mapping:
///
/// - **Opus**: granules are always 48 kHz samples and the stream begins with `pre_skip`
///   samples of decoder priming, which §4 of RFC 7845 says to subtract.
/// - **Vorbis**: granules are PCM samples at the identification header's rate, priming
///   already accounted for.
/// - **FLAC-in-Ogg**: the mapping's bos packet embeds a whole native `STREAMINFO`, so the
///   exact sample count is available without reading the tail at all; the granule (a sample
///   count at the same rate) answers only when STREAMINFO says "unknown".
///
/// `head` must hold the whole first page — [`crate::head::OGG_HEAD_LEN`] — and `tail` the
/// last [`crate::head::OGG_TAIL_LEN`] bytes. `None` duration for a mapping with no
/// identification header here (Theora, Speex, anything unrecognised), or a file whose ends
/// do not parse; the index is the proportional fallback either way, which the duration is
/// what unlocks.
pub fn ogg_seek_index(head: &[u8], tail: &[u8], file_len: u64) -> SeekInfo {
    SeekInfo::proportional(file_len, ogg_duration_ns(head, tail))
}

/// The bos-page magic of the xiph "Ogg Mapping for FLAC" (§mapping): a `0x7F` byte then
/// `"FLAC"`, followed by the mapping version, the header-packet count, and then a whole
/// native FLAC stream head starting at `"fLaC"` — nine bytes in.
const OGG_FLAC_MAGIC: [u8; 5] = [0x7F, b'F', b'L', b'A', b'C'];
/// Byte offset of the embedded `"fLaC"` marker within that bos packet.
const OGG_FLAC_STREAM_HEAD: usize = 9;

/// See [`ogg_seek_index`]. Split out so the mapping arithmetic is testable on byte windows.
fn ogg_duration_ns(head: &[u8], tail: &[u8]) -> Option<u64> {
    // The bos page is the first page of the file; `parse` verifies its CRC, so a head too
    // short to hold the whole page is rejected rather than half-believed.
    let bos = pf_ogg::PageHeader::parse(head).ok()?;
    let packet = bos.payload();
    // Restricted to this logical bitstream: in a grouped (multiplexed) file the last page
    // overall may belong to a different stream than the one being timed (RFC 3533 §4).
    let last = || pf_ogg::last_granule(tail, Some(bos.serial()));

    if let Some(id) = pf_ogg::parse_opus_head(packet) {
        return Some(pf_ogg::opus_duration_ns(last()?, id.pre_skip));
    }
    if let Some(id) = pf_ogg::parse_vorbis_ident(packet) {
        return pf_ogg::vorbis_duration_ns(last()?, id.sample_rate);
    }
    if packet.starts_with(&OGG_FLAC_MAGIC) {
        let info = pf_flac::tags::stream_info(packet.get(OGG_FLAC_STREAM_HEAD..)?)?;
        if info.total_samples > 0 {
            return samples_to_ns(info.total_samples, info.sample_rate);
        }
        return samples_to_ns(last()?, info.sample_rate);
    }
    None
}

/// Nanoseconds for `samples` at `rate` Hz — the sample-grid conversion every duration here
/// goes through. Widened to `u128`: FLAC's 36-bit sample count times 10⁹ leaves a `u64`
/// long before the quotient does. `None` for a zero rate, which no identification header
/// may declare but untrusted bytes can.
fn samples_to_ns(samples: u64, rate: u32) -> Option<u64> {
    if rate == 0 {
        return None;
    }
    let ns = u128::from(samples) * 1_000_000_000 / u128::from(rate);
    Some(ns.min(u128::from(u64::MAX)) as u64)
}

/// Append `(time_ns, byte)` to an index under construction, but only if it *advances* the
/// map, and only if the byte is inside the file.
///
/// [`SeekIndex::resolve`] floors over ascending times, so a non-ascending entry is at best
/// unreachable; a byte offset that went backwards would make a later seek land earlier than
/// an earlier one, which is worse than coarse — it is wrong. A real encoder's table is
/// monotonic in both, and this is what keeps a corrupt one from yielding a nonsense index
/// instead of simply a shorter one.
fn push_ascending(entries: &mut Vec<(u64, u64)>, time_ns: u64, byte: u64, file_len: u64) {
    if byte >= file_len {
        return;
    }
    match entries.last() {
        Some(&(t, b)) if time_ns <= t || byte < b => {}
        _ => entries.push((time_ns, byte)),
    }
}

/// Fill `entries` with a hundred-point straight line: time 0 at byte `start`, and the full
/// `duration_ns` at `start + span`. That is the constant-bitrate assumption — exactly right
/// for a CBR stream, and the same guess the proportional fallback makes for any other, but
/// anchored on the audio rather than on byte 0. A hundred points matches the Xing TOC's
/// density, so a file with a table and one without behave the same way at the seek bar.
fn linear_ramp(
    entries: &mut Vec<(u64, u64)>,
    duration_ns: u64,
    start: u64,
    span: u64,
    file_len: u64,
) {
    const POINTS: u128 = 100;
    for i in 0..POINTS {
        let time = (i * u128::from(duration_ns) / POINTS) as u64;
        let byte = start.saturating_add((i * u128::from(span) / POINTS) as u64);
        push_ascending(entries, time, byte, file_len);
    }
}

/// No keyframe index: the proportional-or-nothing fallback for Ogg and elementary streams.
/// `duration_ns` is whatever the caller could learn (usually `None` pre-decode) — with it
/// and `file_len`, [`SeekIndex::resolve`] estimates a byte offset; without it, time seeking
/// is simply unavailable and the CLI reports so.
pub fn proportional(file_len: u64, duration_ns: Option<u64>) -> SeekInfo {
    SeekInfo::proportional(file_len, duration_ns)
}

/// A bounded positional read of `len` bytes at `off`. App-side controller IO (building the
/// seek index before the pipeline), so blocking `read_exact_at` is sanctioned — see the
/// module docs and `clippy.toml`.
#[allow(clippy::disallowed_methods)] // app-side Cues pread before the pipeline; see module docs + clippy.toml
fn pread(path: &str, off: u64, len: usize) -> Option<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let mut buf = vec![0u8; len];
    let f = std::fs::File::open(path).ok()?;
    f.read_exact_at(&mut buf, off).ok()?;
    Some(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use profluens_core::time::Timestamp;

    #[test]
    fn proportional_resolves_by_bytes_when_duration_known() {
        let info = proportional(10_000, Some(10_000_000_000)); // 10 s, 10 kB
        // Halfway in time → halfway in bytes.
        let (byte, landed) =
            info.index.resolve(Timestamp(5_000_000_000), Timestamp(10_000_000_000)).unwrap();
        assert_eq!(byte, 5_000);
        assert_eq!(landed, Timestamp(5_000_000_000), "proportional lands at the request");
    }

    #[test]
    fn proportional_has_no_mapping_without_duration() {
        let info = proportional(10_000, None);
        assert!(info.index.resolve(Timestamp(1_000_000_000), Timestamp::NONE).is_none());
    }

    // --- Elementary streams -----------------------------------------------------------
    //
    // Fixtures are hand-built: an MPEG frame with a real header and a planted Xing block, a
    // FLAC metadata chain, a RIFF/WAVE header, Ogg pages written by `pf-ogg`'s own muxer.
    // Every one of them is the shape a real encoder writes, so the assertions below are
    // about *this* code's arithmetic and anchoring, not about the codec crates' parsers
    // (which have their own tests).

    /// Length of an MPEG-1 Layer III frame at 44.1 kHz / 128 kbit/s: 144 × 128000 ÷ 44100.
    const FRAME_LEN: usize = 417;

    /// `count` MPEG-1 Layer III stereo 128 kbit/s 44.1 kHz frames, with `tag` planted in
    /// the first where the Xing block goes (4 header bytes + 32 bytes of stereo side info).
    ///
    /// At least two frames, always: the framer confirms a header by checking that the next
    /// frame's sync word lands exactly at its end, so a lone frame followed by padding is
    /// not a frame as far as `probe_props` is concerned — which is the correct behaviour,
    /// and the reason this builder exists rather than a single-frame one.
    fn mp3_frames(count: usize, tag: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for i in 0..count.max(2) {
            let mut f = vec![0xFF, 0xFB, 0x90, 0x00];
            f.resize(FRAME_LEN, 0);
            if i == 0 {
                f[36..36 + tag.len()].copy_from_slice(tag);
            }
            out.extend_from_slice(&f);
        }
        out
    }

    /// A Xing block with a frame count and a 100-point TOC (flags `$5` — count + TOC, no
    /// byte count, so the walk cannot pass by luck on a fixed offset).
    fn xing_with_toc(frames: u32, toc: &[u8; 100]) -> Vec<u8> {
        let mut b = b"Xing".to_vec();
        b.extend_from_slice(&5u32.to_be_bytes());
        b.extend_from_slice(&frames.to_be_bytes());
        b.extend_from_slice(toc);
        b
    }

    /// An ID3v2.3 tag of `body` bytes of padding — a 10-byte header with a syncsafe size.
    fn id3v2(body: usize) -> Vec<u8> {
        let mut t = b"ID3\x03\x00\x00".to_vec();
        let n = body as u32;
        t.extend_from_slice(&[
            ((n >> 21) & 0x7F) as u8,
            ((n >> 14) & 0x7F) as u8,
            ((n >> 7) & 0x7F) as u8,
            (n & 0x7F) as u8,
        ]);
        t.resize(10 + body, 0);
        t
    }

    /// A 128-byte ID3v1 block.
    fn id3v1() -> Vec<u8> {
        let mut t = b"TAG".to_vec();
        t.resize(128, 0);
        t
    }

    /// An APEv2 tag holding no items: a 32-byte header and a 32-byte footer, whose
    /// `tag size` field counts the footer plus the (empty) item area.
    fn ape_tag() -> Vec<u8> {
        let mut out = Vec::new();
        for header in [true, false] {
            out.extend_from_slice(b"APETAGEX");
            out.extend_from_slice(&2000u32.to_le_bytes()); // version
            out.extend_from_slice(&32u32.to_le_bytes()); // tag size: footer only
            out.extend_from_slice(&0u32.to_le_bytes()); // item count
            // Flags: bit 31 "has header", bit 29 "is the header".
            let flags: u32 = 0x8000_0000 | if header { 0x2000_0000 } else { 0 };
            out.extend_from_slice(&flags.to_le_bytes());
            out.extend_from_slice(&[0u8; 8]); // reserved
        }
        out
    }

    /// `toc[i] = 256 × i / 100` — the table a constant-bitrate stream has, and the one
    /// whose expected byte offsets can be written down by hand.
    fn linear_toc() -> [u8; 100] {
        let mut t = [0u8; 100];
        for (i, e) in t.iter_mut().enumerate() {
            *e = (i * 256 / 100) as u8;
        }
        t
    }

    #[test]
    fn mp3_toc_entries_are_anchored_past_the_id3v2_tag() {
        // 1000 frames of MPEG-1 Layer III at 44.1 kHz = 1000 × 1152 / 44100 s.
        const FRAMES: u32 = 1000;
        let toc = linear_toc();
        let tag = id3v2(500);
        let mut file = tag.clone();
        file.extend_from_slice(&mp3_frames(8, &xing_with_toc(FRAMES, &toc)));
        file.resize(tag.len() + 400_000, 0); // the rest of the audio
        let len = file.len() as u64;

        let info = mp3_seek_index(&file, &file, 0, len);
        let want_ns = u64::from(FRAMES) * 1152 * 1_000_000_000 / 44_100;
        assert_eq!(info.duration_ns, Some(want_ns), "a Xing frame count is exact");

        let audio_start = tag.len() as u64;
        let span = len - audio_start; // no trailing tags in this fixture
        assert_eq!(info.index.entries.len(), 100, "one entry per TOC point");
        // Point 0 is the first audio byte — *past the tag*, which is the whole point: a
        // proportional index over the file would have put t=0 at byte 0.
        assert_eq!(info.index.entries[0], (0, audio_start));
        // Point 50: half the duration, and `toc[50] = 128` → 128/256 of the audio region.
        assert_eq!(info.index.entries[50], (want_ns / 2, audio_start + span / 2));
        assert_eq!(info.index.entries[99], (want_ns * 99 / 100, audio_start + 253 * span / 256));
        assert!(info.index.entries.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1));

        // …and resolving through it lands on a real audio byte, not on the tag.
        let (byte, landed) =
            info.index.resolve(Timestamp(want_ns / 2), Timestamp(want_ns)).expect("mapped");
        assert_eq!(byte, audio_start + span / 2);
        assert_eq!(landed, Timestamp(want_ns / 2));
    }

    #[test]
    fn mp3_trailing_tags_are_not_audio() {
        // A constant-bitrate stream with tags at both ends: the duration is derived from
        // the audio's byte count, so counting an APEv2 + ID3v1 tail as audio would stretch
        // it. 128 kbit/s = 16000 bytes/s.
        let tag = id3v2(100);
        let ape = ape_tag();
        let mut file = tag.clone();
        file.extend_from_slice(&mp3_frames(8, &[])); // no Xing → the CBR estimate
        file.resize(tag.len() + 160_000, 0); // 10 s of audio at 128 kbit/s
        let audio_len = 160_000u64;
        file.extend_from_slice(&ape);
        file.extend_from_slice(&id3v1());
        let len = file.len() as u64;

        assert_eq!(mp3_trailers(&file, 0, len), ape.len() as u64 + 128);
        let info = mp3_seek_index(&file, &file, 0, len);
        assert_eq!(info.duration_ns, Some(audio_len * 8_000_000 / 128), "10 s exactly");
        assert_eq!(info.duration_ns, Some(10_000_000_000));

        // A tail window that does not end at EOF cannot be measured, so it measures
        // nothing rather than mistaking a mid-file byte for a footer.
        assert_eq!(mp3_trailers(&file[..64], 0, len), 0);
        // A short tail window that *does* end at EOF measures the same as the whole file.
        let base = len - 512;
        assert_eq!(mp3_trailers(&file[base as usize..], base, len), ape.len() as u64 + 128);
    }

    #[test]
    fn mp3_without_a_toc_gets_a_ramp_over_the_audio_region() {
        // No Xing block at all: a hundred-point straight line, still anchored past the tag.
        let tag = id3v2(1_000);
        let mut file = tag.clone();
        file.extend_from_slice(&mp3_frames(8, &[]));
        file.resize(tag.len() + 160_000, 0);
        let len = file.len() as u64;

        let info = mp3_seek_index(&file, &file, 0, len);
        assert_eq!(info.duration_ns, Some(10_000_000_000));
        assert_eq!(info.index.entries.len(), 100);
        assert_eq!(info.index.entries[0], (0, tag.len() as u64));
        assert_eq!(info.index.entries[50], (5_000_000_000, tag.len() as u64 + 80_000));
    }

    /// A FLAC metadata chain: `fLaC`, STREAMINFO with a rate and sample count, an optional
    /// SEEKTABLE, and a last PADDING block — so `audio_start` has a definite answer.
    fn flac_head(rate: u32, total_samples: u64, points: &[(u64, u64)], pad: usize) -> Vec<u8> {
        let mut si = [0u8; 34];
        // §8.2 bit packing: rate is 20 bits at bit 80, channels-1 is 3, bits-1 is 5, and
        // the total sample count is the 36 bits from bit 108.
        si[10] = (rate >> 12) as u8;
        si[11] = (rate >> 4) as u8;
        si[12] = ((rate << 4) as u8) | (1 << 1); // channels - 1 = 1 → stereo
        si[13] = (0x0F << 4) | ((total_samples >> 32) as u8 & 0x0F); // bits - 1 = 15 → 16
        si[14..18].copy_from_slice(&(total_samples as u32).to_be_bytes());

        let mut out = b"fLaC".to_vec();
        let mut block = |kind: u8, last: bool, body: &[u8]| {
            out.push(if last { 0x80 | kind } else { kind });
            out.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
            out.extend_from_slice(body);
        };
        block(0, false, &si);
        if !points.is_empty() {
            let mut body = Vec::new();
            for &(sample, offset) in points {
                body.extend_from_slice(&sample.to_be_bytes());
                body.extend_from_slice(&offset.to_be_bytes());
                body.extend_from_slice(&4096u16.to_be_bytes());
            }
            // One placeholder slot (§8.5), which must not become an entry.
            body.extend_from_slice(&u64::MAX.to_be_bytes());
            body.extend_from_slice(&0xDEAD_BEEFu64.to_be_bytes());
            body.extend_from_slice(&0u16.to_be_bytes());
            block(3, false, &body);
        }
        block(1, true, &vec![0u8; pad]);
        out
    }

    #[test]
    fn flac_seektable_entries_are_file_absolute() {
        // A 4-second stream at 44.1 kHz with a point per second.
        let points = [(0u64, 0u64), (44_100, 20_000), (88_200, 41_000), (132_300, 60_500)];
        let head = flac_head(44_100, 44_100 * 4, &points, 64);
        let audio_start = head.len() as u64;
        let len = audio_start + 80_000;

        let info = flac_seek_index(&head, len);
        assert_eq!(info.duration_ns, Some(4_000_000_000), "STREAMINFO is exact");
        // Four real points; the placeholder is not one of them.
        assert_eq!(info.index.entries.len(), 4);
        assert_eq!(info.index.entries[0], (0, audio_start));
        assert_eq!(info.index.entries[1], (1_000_000_000, audio_start + 20_000));
        assert_eq!(info.index.entries[3], (3_000_000_000, audio_start + 60_500));
        assert!(info.index.entries.iter().all(|&(_, b)| b != 0xDEAD_BEEF));

        // Resolving floors to the preceding point and reports where it landed — 2.5 s of
        // requested time comes back as the 2 s seek point.
        let (byte, landed) =
            info.index.resolve(Timestamp(2_500_000_000), Timestamp(4_000_000_000)).unwrap();
        assert_eq!((byte, landed), (audio_start + 41_000, Timestamp(2_000_000_000)));
    }

    #[test]
    fn flac_without_a_seektable_still_reports_its_duration() {
        let head = flac_head(48_000, 48_000 * 90, &[], 16);
        let info = flac_seek_index(&head, head.len() as u64 + 500_000);
        assert_eq!(info.duration_ns, Some(90_000_000_000));
        assert!(info.index.entries.is_empty(), "no table → the proportional fallback");
        // …which is now armed, because the duration is known.
        assert!(info.index.resolve(Timestamp(45_000_000_000), Timestamp(90_000_000_000)).is_some());

        // A zero sample count is STREAMINFO's "unknown" (§8.2), not a zero-length stream.
        let unknown = flac_head(44_100, 0, &[], 0);
        assert_eq!(flac_seek_index(&unknown, 1_000).duration_ns, None);
    }

    #[test]
    fn wav_entries_are_exact_and_frame_aligned() {
        use profluens_audio::{write_pcm_wav, AudioFormat, SampleFormat};
        // 3 seconds of 44.1 kHz stereo s16 — 4 bytes per interchannel frame.
        let format = AudioFormat::new(44_100, 2, SampleFormat::S16);
        let pcm = vec![0u8; 44_100 * 4 * 3];
        let file = write_pcm_wav(&format, &pcm);
        let len = file.len() as u64;
        let data_start = 44u64; // the canonical header

        let info = wav_seek_index(&file, len);
        assert_eq!(info.duration_ns, Some(3_000_000_000));
        // A 100 ms grid over 3 s: 30 entries, the first at the `data` chunk itself.
        assert_eq!(info.index.entries.len(), 30);
        assert_eq!(info.index.entries[0], (0, data_start));
        for &(time, byte) in &info.index.entries {
            let off = byte - data_start;
            assert_eq!(off % 4, 0, "every entry lands on a whole interchannel frame");
            // The entry's time is the *exact* time of its byte, not the grid point.
            assert_eq!(time, off / 4 * 1_000_000_000 / 44_100);
            assert!(byte < len);
        }
        assert_eq!(info.index.entries[15], (1_500_000_000, data_start + 44_100 * 4 * 3 / 2));

        // A long file thins the grid instead of growing the index without bound.
        let hour = write_pcm_wav(&AudioFormat::new(8_000, 1, SampleFormat::U8), &[]);
        let long = wav_seek_index(&hour, 44 + 8_000 * 3_600);
        assert_eq!(long.duration_ns, Some(3_600_000_000_000));
        assert!(long.index.entries.len() <= WAV_MAX_ENTRIES as usize + 1, "{}", long.index.entries.len());
        assert!(long.index.entries.len() > 4_000, "still dense: {}", long.index.entries.len());
    }

    /// One Ogg page, via `pf-ogg`'s own writer so the CRC is real.
    fn ogg_page(flags: u8, granule: u64, serial: u32, seq: u32, body: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        pf_ogg::write_page(&mut v, flags, granule, serial, seq, &[body.len() as u8], body);
        v
    }

    #[test]
    fn ogg_duration_follows_the_codec_mapping() {
        const SERIAL: u32 = 0xABCD;
        // --- Opus (RFC 7845): 48 kHz granules, minus the pre-skip.
        let mut opus = b"OpusHead".to_vec();
        opus.extend_from_slice(&[1, 2]); // version, channels
        opus.extend_from_slice(&312u16.to_le_bytes()); // pre-skip
        opus.extend_from_slice(&44_100u32.to_le_bytes()); // original input rate
        opus.extend_from_slice(&0i16.to_le_bytes());
        opus.push(0);
        let head = ogg_page(0x02, 0, SERIAL, 0, &opus);
        let tail = ogg_page(0x04, 48_000 * 90 + 312, SERIAL, 9, b"audio");
        assert_eq!(ogg_duration_ns(&head, &tail), Some(90_000_000_000));
        // The *input* rate never enters it — a 44.1 kHz source is still counted at 48 kHz.

        // --- Vorbis: PCM samples at the identification header's own rate, no pre-skip.
        let mut vorbis = vec![0x01];
        vorbis.extend_from_slice(b"vorbis");
        vorbis.extend_from_slice(&0u32.to_le_bytes()); // version
        vorbis.push(2); // channels
        vorbis.extend_from_slice(&44_100u32.to_le_bytes()); // rate
        vorbis.extend_from_slice(&[0u8; 16]); // bitrates + blocksizes + framing
        let head = ogg_page(0x02, 0, SERIAL, 0, &vorbis);
        let tail = ogg_page(0x04, 44_100 * 30, SERIAL, 9, b"audio");
        assert_eq!(ogg_duration_ns(&head, &tail), Some(30_000_000_000));

        // --- FLAC-in-Ogg: the bos packet embeds a whole native stream head, so STREAMINFO
        // answers exactly and the tail is not even consulted.
        let mut oggflac = vec![0x7F, b'F', b'L', b'A', b'C', 1, 0, 0, 1];
        oggflac.extend_from_slice(&flac_head(48_000, 48_000 * 7, &[], 0));
        let head = ogg_page(0x02, 0, SERIAL, 0, &oggflac);
        assert_eq!(ogg_duration_ns(&head, &[]), Some(7_000_000_000));

        // --- A mapping with no identification header here (Theora) reports nothing.
        let mut theora = vec![0x80];
        theora.extend_from_slice(b"theora");
        let head = ogg_page(0x02, 0, SERIAL, 0, &theora);
        assert_eq!(ogg_duration_ns(&head, &tail), None);
    }

    #[test]
    fn ogg_last_granule_is_taken_from_the_right_logical_stream() {
        // A grouped file (RFC 3533 §4) whose *video* stream ends last: timing the audio off
        // the final page overall would report the video's length.
        const AUDIO: u32 = 0x0A;
        const VIDEO: u32 = 0x0B;
        let mut opus = b"OpusHead".to_vec();
        opus.extend_from_slice(&[1, 2]);
        opus.extend_from_slice(&0u16.to_le_bytes());
        opus.extend_from_slice(&48_000u32.to_le_bytes());
        opus.extend_from_slice(&0i16.to_le_bytes());
        opus.push(0);
        let head = ogg_page(0x02, 0, AUDIO, 0, &opus);
        let mut tail = ogg_page(0, 48_000 * 10, AUDIO, 8, b"aud");
        tail.extend_from_slice(&ogg_page(0x04, 90_000 * 60, VIDEO, 8, b"vid"));
        assert_eq!(ogg_duration_ns(&head, &tail), Some(10_000_000_000));
    }

    #[test]
    fn hostile_windows_never_panic_and_never_lie() {
        // Every truncation of every fixture, through every builder: no panic, and no index
        // entry that points outside the file.
        let toc = linear_toc();
        let mut mp3 = id3v2(64);
        mp3.extend_from_slice(&mp3_frames(4, &xing_with_toc(1000, &toc)));
        mp3.resize(mp3.len() + 4_000, 0);
        mp3.extend_from_slice(&ape_tag());
        mp3.extend_from_slice(&id3v1());
        let flac = flac_head(44_100, 44_100 * 4, &[(0, 0), (44_100, u64::MAX - 4)], 32);
        let wav = profluens_audio::write_pcm_wav(
            &profluens_audio::AudioFormat::new(44_100, 2, profluens_audio::SampleFormat::S16),
            &vec![0u8; 4_000],
        );
        let ogg = {
            let mut h = b"OpusHead".to_vec();
            h.extend_from_slice(&[1, 2]);
            h.extend_from_slice(&312u16.to_le_bytes());
            h.extend_from_slice(&48_000u32.to_le_bytes());
            h.extend_from_slice(&0i16.to_le_bytes());
            h.push(0);
            ogg_page(0x02, 0, 7, 0, &h)
        };
        let garbage: Vec<u8> = (0..2_000).map(|i| (i as u8).wrapping_mul(31) | 1).collect();

        for src in [&mp3, &flac, &wav, &ogg, &garbage] {
            for n in 0..=src.len() {
                let w = &src[..n];
                let len = src.len() as u64;
                for info in [
                    mp3_seek_index(w, w, 0, len),
                    mp3_seek_index(w, w, len.saturating_sub(n as u64), len),
                    flac_seek_index(w, len),
                    wav_seek_index(w, len),
                    ogg_seek_index(w, w, len),
                ] {
                    for &(_, byte) in &info.index.entries {
                        assert!(byte < len, "an entry outside the file at cut {n}");
                    }
                    assert!(
                        info.index.entries.windows(2).all(|p| p[0].0 < p[1].0),
                        "entries must stay ascending at cut {n}"
                    );
                }
            }
        }
        // A zero-length file is not a division by zero.
        assert_eq!(mp3_seek_index(&[], &[], 0, 0).duration_ns, None);
        assert_eq!(flac_seek_index(&[], 0).duration_ns, None);
        assert_eq!(wav_seek_index(&[], 0).duration_ns, None);
        assert_eq!(ogg_seek_index(&[], &[], 0).duration_ns, None);
    }

    #[test]
    fn empty_entries_means_proportional_fallback() {
        // The MKV cue-less / MP4 no-video path both produce an empty-entries index; assert
        // it behaves as the proportional fallback (not a keyframe floor lookup).
        let info = SeekInfo::proportional(2_000, Some(4_000_000_000));
        assert!(info.index.entries.is_empty());
        let (byte, _) =
            info.index.resolve(Timestamp(2_000_000_000), Timestamp(4_000_000_000)).unwrap();
        assert_eq!(byte, 1_000);
    }
}
