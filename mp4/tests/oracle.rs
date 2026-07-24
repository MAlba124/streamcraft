//! Oracle cross-validation: our hand-written [`Mp4Reader`] sample enumeration is asserted
//! against **oxideav-mp4** (a dev-dependency, per the container rule — never a production
//! dep) over the committed fixtures, plus a second-opinion check against a live `ffprobe`
//! that skips if ffmpeg is absent.
//!
//! The load-bearing property: our sample table resolution — per-sample **count, byte size,
//! keyframe flag, and the `stts`/`ctts` pts/dts arithmetic** (in media-timescale ticks) —
//! matches an entirely independent MP4 reader, so a table-fold bug (a wrong `stsc` run, an
//! off-by-one `stco`, a mis-signed v1 `ctts`) shows up as a disagreement rather than a
//! silently-wrong demux. oxideav-mp4's `next_packet` serves each sample's raw bytes
//! (offset+size slice out of `mdat`, no reframing) with `pts`=CTS, `dts`=DTS, and the sync
//! flag — the same values our resolver computes, so the comparison is direct.
//!
//! Fixtures (see `tests/fixtures/GENERATION.md`):
//! - `tiny_h264.mp4` — Baseline, no B-frames, **no `ctts`** (pts == dts).
//! - `bframes_h264.mp4` — Main, 2 B-frames, **`ctts` present** (composition offsets).

use std::io::Cursor;
use std::path::PathBuf;

use oxideav_core::{Error as OxError, NullCodecResolver};

use sc_mp4::Mp4Reader;

/// Path to a committed fixture.
fn fixture(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

/// The whole fixture bytes.
fn fixture_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|e| panic!("read fixture {name}: {e}"))
}

/// The file head through the end of `moov` (what `Mp4Reader::new` / `Mp4Demux::new` need).
/// For a `+faststart` file `moov` precedes `mdat`, so this is the prefix up to `mdat`.
fn head_through_moov(file: &[u8]) -> Vec<u8> {
    // Walk top-level boxes; return the prefix ending at the box after which moov is complete.
    // Simplest robust rule: the head must include ftyp + moov; find mdat and cut before it
    // (faststart guarantees moov is before mdat). If no mdat, the whole file is the head.
    let mut at = 0usize;
    while at + 8 <= file.len() {
        let size = u32::from_be_bytes([file[at], file[at + 1], file[at + 2], file[at + 3]]) as usize;
        let kind = &file[at + 4..at + 8];
        if kind == b"mdat" {
            return file[..at].to_vec();
        }
        let advance = if size == 0 { file.len() - at } else if size == 1 {
            // 64-bit largesize
            u64::from_be_bytes(file[at + 8..at + 16].try_into().unwrap()) as usize
        } else {
            size
        };
        if advance == 0 {
            break;
        }
        at += advance;
    }
    file.to_vec()
}

/// One oracle packet (raw sample) from oxideav-mp4, reduced to the fields we compare.
#[derive(Debug, PartialEq, Eq)]
struct OraclePacket {
    stream_index: u32,
    size: usize,
    pts: i64,
    dts: i64,
    keyframe: bool,
}

/// Enumerate every sample of `file` via oxideav-mp4's demuxer, plus each stream's timescale.
/// Returns `(packets_in_file_order, per_stream_timescale)`.
fn oxideav_enumerate(file: &[u8]) -> (Vec<OraclePacket>, Vec<u32>) {
    let input = Box::new(Cursor::new(file.to_vec()));
    let mut demux = oxideav_mp4::demux::open(input, &NullCodecResolver).expect("oxideav open");
    // Time base den == media timescale (time_base = 1/rate).
    let timescales: Vec<u32> = demux
        .streams()
        .iter()
        .map(|s| s.time_base.den().max(0) as u32)
        .collect();
    let mut packets = Vec::new();
    loop {
        match demux.next_packet() {
            Ok(pkt) => packets.push(OraclePacket {
                stream_index: pkt.stream_index,
                size: pkt.data.len(),
                pts: pkt.pts.expect("mp4 sample has a pts"),
                dts: pkt.dts.expect("mp4 sample has a dts"),
                keyframe: pkt.flags.keyframe,
            }),
            Err(OxError::Eof) => break,
            Err(e) => panic!("oxideav next_packet: {e:?}"),
        }
    }
    (packets, timescales)
}

/// Cross-validate our resolver against oxideav-mp4 for a fixture: identical sample count,
/// and per sample (matched by file order within a track) identical size / pts / dts /
/// keyframe, all in media-timescale ticks.
fn assert_agrees_with_oxideav(name: &str) {
    let file = fixture_bytes(name);
    let head = head_through_moov(&file);

    // Our resolution.
    let ours = Mp4Reader::new(&head).unwrap_or_else(|e| panic!("our resolve {name}: {e:?}"));
    let our_timescales: Vec<u32> = ours.tracks().iter().map(|t| t.timescale).collect();

    // Oracle.
    let (oracle, oracle_timescales) = oxideav_enumerate(&file);

    assert_eq!(
        ours.samples().len(),
        oracle.len(),
        "{name}: sample count must match oxideav ({} vs {})",
        ours.samples().len(),
        oracle.len()
    );
    assert_eq!(
        our_timescales, oracle_timescales,
        "{name}: per-track media timescale must match oxideav"
    );

    // Both lists are in file-offset order across all tracks; compare position by position.
    // Our samples sort by offset; oxideav serves in offset order too, so index i pairs up.
    for (i, (o, x)) in ours.samples().iter().zip(oracle.iter()).enumerate() {
        assert_eq!(
            o.track_index as u32, x.stream_index,
            "{name}: sample {i} track/stream index"
        );
        assert_eq!(o.size as usize, x.size, "{name}: sample {i} size (mdat slice length)");
        assert_eq!(o.pts, x.pts, "{name}: sample {i} pts (dts + ctts - edit_shift, ticks)");
        assert_eq!(o.dts, x.dts, "{name}: sample {i} dts (stts sum - edit_shift, ticks)");
        assert_eq!(o.sync, x.keyframe, "{name}: sample {i} sync/keyframe flag");
    }
}

/// Baseline fixture (no ctts): every sample agrees with oxideav; pts == dts throughout.
#[test]
fn tiny_h264_matches_oxideav() {
    assert_agrees_with_oxideav("tiny_h264.mp4");
    // Extra: with no ctts, our pts must equal dts for every sample.
    let file = fixture_bytes("tiny_h264.mp4");
    let ours = Mp4Reader::new(&head_through_moov(&file)).unwrap();
    assert!(
        ours.samples().iter().all(|s| s.pts == s.dts),
        "no ctts box → pts == dts for every sample"
    );
    // And exactly one sync sample (a single closed GOP of 15 with keyint=15).
    let syncs = ours.samples().iter().filter(|s| s.sync).count();
    assert_eq!(syncs, 1, "one keyframe (the single IDR)");
    assert_eq!(ours.samples().len(), 15, "15 frames");
}

/// B-frame fixture (ctts present): the composition-offset arithmetic agrees with oxideav,
/// and at least one sample has pts != dts (the reorder the ctts encodes).
#[test]
fn bframes_h264_matches_oxideav() {
    assert_agrees_with_oxideav("bframes_h264.mp4");
    let file = fixture_bytes("bframes_h264.mp4");
    let ours = Mp4Reader::new(&head_through_moov(&file)).unwrap();
    assert!(
        ours.samples().iter().any(|s| s.pts != s.dts),
        "ctts box present → some sample has pts != dts (B-frame reorder)"
    );
}

/// Second-opinion oracle: a live `ffprobe` frame dump. Skips if ffmpeg/ffprobe is absent
/// (the committed fixtures are frozen — this does not regenerate them). Asserts the frame
/// count and keyframe count match ffprobe's view.
#[test]
fn ffprobe_second_opinion() {
    let Some(ffprobe) = which("ffprobe") else {
        eprintln!("ffprobe absent — skipping the live-oracle second opinion");
        return;
    };
    for name in ["tiny_h264.mp4", "bframes_h264.mp4"] {
        let path = fixture(name);
        // `-show_frames` with a compact CSV of key_frame flags for the video stream.
        let out = std::process::Command::new(&ffprobe)
            .args([
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "frame=key_frame",
                "-of", "csv=p=0",
            ])
            .arg(&path)
            .output()
            .expect("run ffprobe");
        assert!(out.status.success(), "ffprobe failed on {name}: {:?}", out.status);
        let text = String::from_utf8_lossy(&out.stdout);
        // Each line is a `key_frame` value (`0`/`1`); ffprobe 8.x may append a stray trailing
        // comma to the first field, so take the leading digit of each non-empty line.
        let flags: Vec<char> = text
            .lines()
            .filter_map(|l| l.trim().chars().next())
            .filter(|c| c.is_ascii_digit())
            .collect();
        let ff_frames = flags.len();
        let ff_keys = flags.iter().filter(|&&c| c == '1').count();

        let file = fixture_bytes(name);
        let ours = Mp4Reader::new(&head_through_moov(&file)).unwrap();
        assert_eq!(ours.samples().len(), ff_frames, "{name}: frame count vs ffprobe");
        let our_keys = ours.samples().iter().filter(|s| s.sync).count();
        assert_eq!(our_keys, ff_keys, "{name}: keyframe count vs ffprobe");
        eprintln!("ffprobe agrees on {name}: {ff_frames} frames, {ff_keys} keyframes");
    }
}

/// Locate an executable on PATH (a tiny `which`).
fn which(exe: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let cand = dir.join(exe);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}
