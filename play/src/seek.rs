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
//! - **Ogg / elementary**: no index; the proportional fallback (or none, when the duration
//!   is unknown) applies.
//!
//! This is controller code running before the pipeline; the two file reads it needs
//! (Cues pread, whole-head reparse) carry the documented `#[allow]` per `clippy.toml`.

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
