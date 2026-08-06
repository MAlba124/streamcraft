//! `probe` — dump the tracks an AVI file demuxes to, and validate the parse by walking the
//! whole `movi` list (chunk counts + bytes per stream) and, when present, the `idx1` index.
//!
//! ```text
//! cargo run --release -p pf-avi --example probe -- FILE.avi
//! ```
//!
//! This is **app/tooling code**, so the file IO here is the sanctioned exception to the
//! reactor-only rule (clippy.toml header): the demuxer *element* never opens a file — it is
//! fed bytes on its sink pad — but a probe tool legitimately reads the file directly, with an
//! `#[allow(clippy::disallowed_methods)]` per read. It mirrors how `sdl3/examples/play_file.rs`
//! preads the MKV Cues range app-side to build a `SeekIndex`.

use pf_avi::riff::{self, MoviWalker, StreamKind};
use std::io::Read;

/// The stream head: enough of the file to cover the `RIFF/AVI` header + `LIST 'hdrl'` + the
/// `movi` list header. 1 MiB is generous — real files put `hdrl` first, well under 64 KiB.
const HEADER_PREFIX: usize = 1024 * 1024;

/// Read the leading `n` bytes of `path` (app-side tooling — the sanctioned reactor exception).
#[allow(clippy::disallowed_methods)] // probe tooling reads the file directly; the element does not
fn read_prefix(path: &str, n: usize) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; n];
    let got = f.read(&mut buf)?;
    buf.truncate(got);
    Ok(buf)
}

/// Read `len` bytes at absolute offset `off` (for the trailing `idx1`). App-side tooling.
#[allow(clippy::disallowed_methods)] // probe tooling reads the file directly; the element does not
fn read_at(path: &str, off: u64, len: usize) -> std::io::Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; len];
    let got = f.read_at(&mut buf, off)?;
    buf.truncate(got);
    Ok(buf)
}

/// Walk the whole `movi` list from the file, counting chunks and bytes per stream. Reads the
/// file in bounded blocks and feeds the streaming [`MoviWalker`] — the exact byte path the
/// element sees, so the counts here are what the demuxer would emit. App-side tooling.
#[allow(clippy::disallowed_methods)] // probe tooling reads the file directly; the element does not
fn walk_movi(path: &str, movi_data_start: u64, nstreams: usize) -> std::io::Result<Vec<(u64, u64)>> {
    use std::io::{Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    f.seek(SeekFrom::Start(movi_data_start))?;
    let mut walker = MoviWalker::new();
    let mut counts = vec![(0u64, 0u64); nstreams + 8]; // (chunks, bytes); slack for stray ids
    let mut block = vec![0u8; 4 * 1024 * 1024];
    loop {
        let got = f.read(&mut block)?;
        if got == 0 {
            break;
        }
        walker.push(&block[..got]);
        while let Some(chunk) = walker.next_chunk() {
            if chunk.stream_index < counts.len() {
                counts[chunk.stream_index].0 += 1;
                counts[chunk.stream_index].1 += chunk.data.len() as u64;
            }
        }
    }
    counts.truncate(nstreams);
    Ok(counts)
}

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: probe FILE.avi");
            std::process::exit(2);
        }
    };
    let file_len = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    // 1) Discover streams from the header prefix (the same probe the element runs in preroll).
    let header = read_prefix(&path, HEADER_PREFIX).expect("read header");
    let (avi, movi) = match riff::probe_header(&header) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("not a parseable AVI: {e:?}");
            std::process::exit(1);
        }
    };
    let movi = movi.expect("movi located in the 1 MiB prefix");

    println!("file: {path}  ({file_len} bytes)");
    println!(
        "movi data starts at byte {}  |  OpenDML header present: {}",
        movi.movi_data_start, avi.has_odml
    );
    println!("{} stream(s):", avi.streams.len());
    for s in &avi.streams {
        let family = pf_avi::codec::family_for(s);
        match s.kind {
            StreamKind::Video => {
                let fps = if s.scale != 0 { s.rate as f64 / s.scale as f64 } else { 0.0 };
                println!(
                    "  [{}] VIDEO family={family} fourcc={} {}x{}  {:.3} fps  declared frames={}",
                    s.index,
                    String::from_utf8_lossy(&s.handler),
                    s.width,
                    s.height,
                    fps,
                    s.length,
                );
            }
            StreamKind::Audio => {
                println!(
                    "  [{}] AUDIO family={family} tag=0x{:04x} {} Hz  {} ch  {} bits  sample_size={}",
                    s.index, s.format_tag, s.samples_per_sec, s.channels, s.bits_per_sample, s.sample_size,
                );
            }
            StreamKind::Other => {
                println!("  [{}] OTHER family={family} handler={}", s.index, String::from_utf8_lossy(&s.handler));
            }
        }
    }
    if let Some(ns) = avi.duration_ns() {
        println!("duration (from header): {:.3} s", ns as f64 / 1e9);
    }

    // 2) Walk the whole movi list to get the real per-stream chunk/byte counts (what the
    //    demuxer would emit). This is the authoritative frame count to compare against ffprobe.
    println!("\nwalking movi (this reads the whole file)…");
    let counts = walk_movi(&path, movi.movi_data_start, avi.streams.len()).expect("walk movi");
    for s in &avi.streams {
        let (chunks, bytes) = counts.get(s.index).copied().unwrap_or((0, 0));
        let kind = match s.kind {
            StreamKind::Video => "video",
            StreamKind::Audio => "audio",
            StreamKind::Other => "other",
        };
        println!("  stream {} ({kind}): {chunks} chunks, {bytes} bytes", s.index);
    }

    // 3) idx1 → SeekIndex demo (spec: flush/seek). Find the idx1 chunk at the file tail: it is
    //    a top-level chunk after movi, so scan the last stretch of the file for its id.
    if let Some((idx1_off, idx1_size)) = find_idx1(&path, file_len) {
        let payload = read_at(&path, idx1_off + 8, idx1_size as usize).expect("read idx1");
        if let Some(si) = pf_avi::build_seek_index(&header, &payload, file_len) {
            println!(
                "\nidx1 at byte {idx1_off} ({idx1_size} bytes) → SeekIndex with {} video keyframe entries",
                si.entries.len()
            );
            if let (Some(first), Some(last)) = (si.entries.first(), si.entries.last()) {
                println!(
                    "  first keyframe: {:.3} s → byte {}   last: {:.3} s → byte {}",
                    first.0 as f64 / 1e9,
                    first.1,
                    last.0 as f64 / 1e9,
                    last.1,
                );
            }
        }
    } else {
        println!("\nno idx1 found (would rely on the proportional seek fallback + resync)");
    }
}

/// Locate the top-level `idx1` chunk near the file tail by scanning backwards for its id.
/// The `idx1` is written after `movi` (AVI RIFF Reference), so a bounded tail scan finds it
/// without walking the whole 2 GB. App-side tooling.
#[allow(clippy::disallowed_methods)] // probe tooling reads the file directly; the element does not
fn find_idx1(path: &str, file_len: u64) -> Option<(u64, u32)> {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path).ok()?;
    // idx1 for a 2 GB file is a few MB; scan a generous tail window for the id, then read its
    // size. We search the last 16 MiB (covers a very large index and the trailing JUNK).
    let window = (16 * 1024 * 1024).min(file_len) as usize;
    let start = file_len - window as u64;
    let mut buf = vec![0u8; window];
    let got = f.read_at(&mut buf, start).ok()?;
    buf.truncate(got);
    // Find the *last* plausible `idx1` id whose declared size fits before EOF.
    let mut found = None;
    let mut i = 0;
    while i + 8 <= buf.len() {
        if &buf[i..i + 4] == b"idx1" {
            let size = u32::from_le_bytes([buf[i + 4], buf[i + 5], buf[i + 6], buf[i + 7]]);
            let abs = start + i as u64;
            if abs + 8 + size as u64 <= file_len {
                found = Some((abs, size));
            }
        }
        i += 1;
    }
    found
}
