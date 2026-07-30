//! Structural round-trip for the Matroska muxer (spec: `mkv/spec/MATROSKA.md`).
//!
//! The load-bearing property: bytes built by [`MatroskaWriter`] read back — with the
//! minimal in-crate EBML reader in `common/` — to the exact structure they encode. We
//! assert the EBML Header is present, each TrackEntry carries `CodecID == "A_FLAC"` and the
//! expected `CodecPrivate`, and every Cluster's SimpleBlocks carry the input frames with the
//! right track number and monotonically non-decreasing absolute timestamps. Frames are real
//! `A_FLAC` frames produced by the hand-written `pf-flac` encoder, so the whole chain is
//! exercised without any external tool.
//!
//! An OPTIONAL `ffmpeg` oracle cross-validates the file if the binary is present, and
//! **skips gracefully** if not (the project sanctions a system `ffmpeg` only as a dev-time
//! oracle, never linked).

mod common;

use common::{parse_simple_block, walk, SimpleBlock};
use pf_flac::{FlacEncoder, SampleFormat};
use pf_mkv::ebml::id;
use pf_mkv::{MatroskaWriter, TrackConfig};

/// Every master ID our muxer emits — the reader descends into these (spec `§ID-tree`).
const MASTERS: &[&[u8]] = &[
    id::EBML,
    id::SEGMENT,
    id::INFO,
    id::TRACKS,
    id::TRACK_ENTRY,
    id::AUDIO,
    id::CLUSTER,
];

/// Produce a real native FLAC stream head (`fLaC` + finalised STREAMINFO) plus a list of
/// FLAC frames for `n_frames` blocks of `block` interchannel samples of S16 audio. Returns
/// `(codec_private, frames)`. Uses the hand-written `pf-flac` encoder so the bytes are
/// genuine, back-patched STREAMINFO and real frame data.
fn make_flac(sample_rate: u32, channels: u32, block: usize, n_frames: usize) -> (Vec<u8>, Vec<Vec<u8>>) {
    let (mut enc, mut header) = FlacEncoder::new(sample_rate, channels, SampleFormat::S16).expect("encoder");
    let mut frames = Vec::new();
    let mut phase = 0i32;
    for _ in 0..n_frames {
        // A block of interleaved S16 PCM (a simple ramp/tone — content is irrelevant).
        let mut pcm = Vec::with_capacity(block * channels as usize * 2);
        for _ in 0..block {
            for c in 0..channels {
                let v = ((phase.wrapping_mul(37) + c as i32 * 1000) & 0x3FFF) as i16 - 0x2000;
                pcm.extend_from_slice(&v.to_le_bytes());
                phase = phase.wrapping_add(1);
            }
        }
        let mut frame = Vec::new();
        enc.encode_interleaved(&pcm, &mut frame).expect("encode");
        // One block ≤ max_block_size → exactly one frame; keep them 1:1 for the test.
        frames.push(frame);
    }
    // Back-patch STREAMINFO with the real min/max sizes + total samples (spec: A_FLAC =
    // native FLAC head). `header` now holds fLaC + the finalised STREAMINFO block.
    let body = enc.finish();
    let off = pf_flac::streaminfo_offset();
    header[off..off + body.len()].copy_from_slice(&body);
    (header, frames)
}

/// Find the first TrackEntry's CodecID string and CodecPrivate bytes in the muxed stream by
/// walking the tree. Returns `(codec_id, codec_private)`.
fn read_first_track(stream: &[u8]) -> (String, Vec<u8>) {
    let els = walk(stream, MASTERS);
    let mut codec_id = None;
    let mut codec_private = None;
    for (_depth, el) in &els {
        if el.id == id::CODEC_ID {
            let (s, e) = el.data.unwrap();
            codec_id = Some(String::from_utf8(stream[s..e].to_vec()).unwrap());
        } else if el.id == id::CODEC_PRIVATE {
            let (s, e) = el.data.unwrap();
            codec_private = Some(stream[s..e].to_vec());
        }
        // Stop after the first track's fields (the first CodecPrivate we complete).
        if codec_id.is_some() && codec_private.is_some() {
            break;
        }
    }
    (codec_id.expect("CodecID present"), codec_private.expect("CodecPrivate present"))
}

/// Collect every SimpleBlock in the stream **with its absolute timestamp** (cluster base +
/// relative), in stream order. Walks Clusters, tracking each Cluster's `Timestamp` child as
/// the base for the blocks that follow it.
fn read_blocks(stream: &[u8]) -> Vec<(i64, SimpleBlock)> {
    let els = walk(stream, MASTERS);
    let mut out = Vec::new();
    let mut cluster_base = 0i64;
    for (_depth, el) in &els {
        if el.id == id::TIMESTAMP {
            let (s, e) = el.data.unwrap();
            // Cluster Timestamp is an EBML uint (big-endian, minimal length).
            let mut v = 0i64;
            for &b in &stream[s..e] {
                v = (v << 8) | b as i64;
            }
            cluster_base = v;
        } else if el.id == id::SIMPLE_BLOCK {
            let (s, e) = el.data.unwrap();
            let blk = parse_simple_block(&stream[s..e]);
            out.push((cluster_base + blk.rel_ts as i64, blk));
        }
    }
    out
}

#[test]
fn single_flac_track_structural_roundtrip() {
    let (codec_private, frames) = make_flac(48_000, 2, 4096, 6);
    let track = TrackConfig::flac(1, codec_private.clone(), 48_000.0, 2, 16);

    let mut w = MatroskaWriter::new(vec![track]);
    let mut out = Vec::new();
    w.write_header(&mut out).unwrap();
    // Space frames by one block (4096 samples @ 48k ≈ 85.333 ms) so several land in one
    // cluster and timestamps are strictly increasing.
    let dur_ns = 4096u64 * 1_000_000_000 / 48_000;
    for (i, f) in frames.iter().enumerate() {
        w.write_frame(&mut out, 1, i as u64 * dur_ns, f, true).unwrap();
    }
    w.finalize(&mut out);

    // --- EBML Header present, at the very start ---
    assert_eq!(&out[..4], id::EBML, "stream starts with the EBML Header ID");
    let els = walk(&out, MASTERS);
    assert!(els.iter().any(|(_, e)| e.id == id::EBML), "EBML Header element present");
    assert!(els.iter().any(|(_, e)| e.id == id::SEGMENT), "Segment present");
    assert!(els.iter().any(|(_, e)| e.id == id::TRACKS), "Tracks present");

    // --- One TrackEntry, CodecID == A_FLAC, CodecPrivate matches exactly ---
    let track_entries = els.iter().filter(|(_, e)| e.id == id::TRACK_ENTRY).count();
    assert_eq!(track_entries, 1, "exactly one TrackEntry");
    let (cid, cpriv) = read_first_track(&out);
    assert_eq!(cid, "A_FLAC", "CodecID is A_FLAC");
    assert_eq!(cpriv, codec_private, "CodecPrivate is the exact fLaC+STREAMINFO blob");
    // Sanity: the CodecPrivate really is a native FLAC head.
    assert_eq!(&cpriv[..4], b"fLaC", "CodecPrivate begins with the fLaC marker");

    // --- Clusters carry the frames: right track, right bytes, monotonic timestamps ---
    let blocks = read_blocks(&out);
    assert_eq!(blocks.len(), frames.len(), "one SimpleBlock per input frame");
    let mut last_ts = i64::MIN;
    for (i, ((abs_ts, blk), frame)) in blocks.iter().zip(frames.iter()).enumerate() {
        assert_eq!(blk.track, 1, "block {i} on track 1");
        assert!(blk.keyframe, "FLAC block {i} flagged keyframe");
        assert_eq!(&blk.frame, frame, "block {i} carries the exact FLAC frame bytes");
        assert!(*abs_ts >= last_ts, "block {i} timestamp monotonically non-decreasing");
        last_ts = *abs_ts;
    }
    // At least one Cluster was emitted.
    assert!(els.iter().any(|(_, e)| e.id == id::CLUSTER), "at least one Cluster");
}

#[test]
fn two_track_library_roundtrip() {
    // The element is single-track, but the writer is N-track: prove 2 tracks at the library
    // level. Track 1 stereo 48k, track 2 mono 44.1k, each with its own real FLAC head.
    let (cp1, frames1) = make_flac(48_000, 2, 4096, 4);
    let (cp2, frames2) = make_flac(44_100, 1, 4096, 4);
    let tracks = vec![
        TrackConfig::flac(1, cp1.clone(), 48_000.0, 2, 16),
        TrackConfig::flac(2, cp2.clone(), 44_100.0, 1, 16),
    ];

    let mut w = MatroskaWriter::new(tracks);
    let mut out = Vec::new();
    w.write_header(&mut out).unwrap();
    // Interleave the two tracks' frames on a shared millisecond timeline.
    let dur = 85_000_000u64; // ~one 48k block, ns
    for i in 0..4 {
        w.write_frame(&mut out, 1, i as u64 * dur, &frames1[i], true).unwrap();
        w.write_frame(&mut out, 2, i as u64 * dur, &frames2[i], true).unwrap();
    }
    w.finalize(&mut out);

    let els = walk(&out, MASTERS);
    // Two TrackEntries with the two CodecIDs.
    let entries = els.iter().filter(|(_, e)| e.id == id::TRACK_ENTRY).count();
    assert_eq!(entries, 2, "two TrackEntries for two tracks");

    // Both CodecPrivates present and distinct (each track's own STREAMINFO).
    let mut cprivs = Vec::new();
    for (_d, el) in &els {
        if el.id == id::CODEC_PRIVATE {
            let (s, e) = el.data.unwrap();
            cprivs.push(out[s..e].to_vec());
        }
    }
    assert_eq!(cprivs.len(), 2, "two CodecPrivate blobs");
    assert!(cprivs.contains(&cp1), "track 1 CodecPrivate present");
    assert!(cprivs.contains(&cp2), "track 2 CodecPrivate present");

    // Blocks for both tracks came through, matching the input frames per track.
    let blocks = read_blocks(&out);
    let t1: Vec<&Vec<u8>> = blocks.iter().filter(|(_, b)| b.track == 1).map(|(_, b)| &b.frame).collect();
    let t2: Vec<&Vec<u8>> = blocks.iter().filter(|(_, b)| b.track == 2).map(|(_, b)| &b.frame).collect();
    assert_eq!(t1.len(), frames1.len(), "all track-1 frames muxed");
    assert_eq!(t2.len(), frames2.len(), "all track-2 frames muxed");
    for (got, want) in t1.iter().zip(frames1.iter()) {
        assert_eq!(*got, want, "track 1 frame bytes preserved");
    }
    for (got, want) in t2.iter().zip(frames2.iter()) {
        assert_eq!(*got, want, "track 2 frame bytes preserved");
    }
}

/// OPTIONAL oracle: if a system `ffmpeg` is on `PATH`, write the muxed file to a temp path
/// and let `ffmpeg` probe it, asserting it recognises a Matroska file with a FLAC audio
/// stream. Skips (passes) cleanly when `ffmpeg` is absent so CI without it still goes green.
#[test]
fn ffmpeg_oracle_if_present() {
    use std::process::Command;

    // Locate ffmpeg; skip gracefully if not installed.
    let ffmpeg = match which_ffmpeg() {
        Some(p) => p,
        None => {
            eprintln!("ffmpeg not found — skipping the optional oracle cross-validation");
            return;
        }
    };

    let (codec_private, frames) = make_flac(48_000, 2, 4096, 8);
    let track = TrackConfig::flac(1, codec_private, 48_000.0, 2, 16);
    let mut w = MatroskaWriter::new(vec![track]);
    let mut out = Vec::new();
    w.write_header(&mut out).unwrap();
    let dur_ns = 4096u64 * 1_000_000_000 / 48_000;
    for (i, f) in frames.iter().enumerate() {
        w.write_frame(&mut out, 1, i as u64 * dur_ns, f, true).unwrap();
    }
    w.finalize(&mut out);

    // Write to a unique temp file.
    let path = std::env::temp_dir().join(format!("pf-mkv-oracle-{}.mkv", std::process::id()));
    std::fs::write(&path, &out).expect("write temp mkv");

    // `ffmpeg -i <file>` prints stream info to stderr and exits non-zero (no output file),
    // which is expected; we assert on the *content* it reports, not the exit code.
    let output = Command::new(&ffmpeg)
        .arg("-hide_banner")
        .arg("-i")
        .arg(&path)
        .output()
        .expect("run ffmpeg");
    let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
    // Keep the file for manual inspection when PF_MKV_KEEP is set; otherwise clean up.
    if std::env::var_os("PF_MKV_KEEP").is_none() {
        let _ = std::fs::remove_file(&path);
    } else {
        eprintln!("PF_MKV_KEEP set — left oracle file at {}", path.display());
    }

    assert!(
        stderr.contains("matroska"),
        "ffmpeg identifies the container as Matroska; stderr was:\n{stderr}"
    );
    assert!(
        stderr.contains("flac"),
        "ffmpeg identifies a FLAC stream; stderr was:\n{stderr}"
    );
}

/// Look for `ffmpeg` on `PATH` without spawning a shell. Returns the first hit, or `None`.
fn which_ffmpeg() -> Option<std::path::PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let cand = dir.join("ffmpeg");
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}
