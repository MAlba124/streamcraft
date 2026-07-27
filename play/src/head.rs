//! Per-container source preparation: the bytes a demuxer's constructor needs before the
//! pipeline exists (spec: no-bins — the controller runs this setup; the demuxer element
//! is fed the *whole* file on its sink pad at run time, but its constructor wants only the
//! header/metadata region up front). This is app-side controller code running *before*
//! any pipeline, so blocking std IO is legitimate — the reactor rule (`ctx.io()`) binds
//! *element* code, and the workspace clippy exception for setup code is exactly this case
//! (see `clippy.toml`'s header). Every std-IO call below carries the documented `#[allow]`
//! with a one-line justification.
//!
//! - MKV: the `MkvDemux::new` header prefix — everything before the first Cluster (EBML
//!   Header + Segment metadata). Lifted from `sdl3/examples/play_file.rs`, keeping its
//!   8 MiB bound.
//! - MP4: the `Mp4Demux::new` head — `ftyp` + the whole `moov`, box-walked with seeks so
//!   a multi-GB file is never slurped. Lifted from `mp4/examples/remux_to_mkv.rs`.
//! - Ogg / FLAC / MP3 / ADTS / WAV: no head — the demuxer/decoder learns everything from
//!   the stream itself, so the controller reads nothing here.

use std::io::{Read, Seek, SeekFrom};

use sc_mkv::ebml::id;

/// Read the first [`probe::PREFIX_LEN`](crate::probe::PREFIX_LEN)-plus bytes for typefind.
/// A single small read; the caller classifies, then chooses the head strategy below.
///
/// Kept separate from the head builders because typefind must run first (it decides the
/// container) and needs only a handful of bytes — no reason to walk `moov` or scan for a
/// Cluster before we even know it's an MP4 or MKV.
#[allow(clippy::disallowed_methods)] // app-side typefind before any pipeline exists; see module docs + clippy.toml
pub fn read_prefix(path: &str, want: usize) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; want];
    let n = f.read(&mut buf)?;
    buf.truncate(n);
    Ok(buf)
}

/// The MKV stream head: everything before the first `Cluster` — the EBML Header plus the
/// Segment metadata (`SeekHead`/`Info`/`Tracks`/`Tags`/…), which is all `MkvDemux::new`
/// needs to discover tracks at preroll (spec: dynamic pads — topology settles at preroll).
///
/// 8 MiB is generous for real files' metadata region (the play_file bound). Truncating at
/// the first Cluster keeps the constructor from holding megabytes of media, and the
/// element re-reads the whole stream from byte 0 at run time regardless.
///
/// Errors (rather than the play_file `expect`) when no Cluster appears in the bound — a
/// truncated or non-Matroska file, which the player should report cleanly, not panic on.
#[allow(clippy::disallowed_methods)] // app-side header read before the pipeline; see module docs + clippy.toml
pub fn mkv_header_prefix(path: &str) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let mut head = vec![0u8; 8 * 1024 * 1024];
    let n = f.read(&mut head)?;
    head.truncate(n);
    // The four-byte Cluster id marks the start of media; the head is everything before it.
    let cluster = head.windows(4).position(|w| w == id::CLUSTER).ok_or_else(|| {
        std::io::Error::other(
            "no Cluster in the first 8 MiB — not a (supported/streamable) Matroska file",
        )
    })?;
    head.truncate(cluster);
    Ok(head)
}

/// The MP4 head: `ftyp` + everything up to (not including) `mdat`, box-walked with seeks so
/// a multi-GB file is never slurped whole (lifted from `mp4/examples/remux_to_mkv.rs`'s
/// `read_head`). `Mp4Demux::new` resolves its sample tables from these bytes at preroll.
///
/// Requires a progressive / `+faststart` layout (`moov` before `mdat`), which is also what
/// streaming the file through the demuxer requires: a `moov`-after-`mdat` file produces a
/// **clear error here** (not a hang), the documented contract for the box-walk.
#[allow(clippy::disallowed_methods)] // app-side box-walk before the pipeline; see module docs + clippy.toml
pub fn mp4_head(path: &str) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let mut at = 0u64;
    loop {
        if at + 8 > len {
            // Walked off the end without finding `mdat`: not a progressive MP4 we can
            // stream (this is where a `moov`-after-`mdat` / fragmented file lands — a clear
            // error, never a hang).
            return Err(std::io::Error::other(
                "no mdat box before EOF — is this a faststart (moov-before-mdat) MP4?",
            ));
        }
        let mut hdr = [0u8; 8];
        f.seek(SeekFrom::Start(at))?;
        f.read_exact(&mut hdr)?;
        let size = u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as u64;
        if &hdr[4..8] == b"mdat" {
            break;
        }
        let advance = match size {
            0 => len - at, // a box with size 0 runs to EOF (ISO/IEC 14496-12 §4.2)
            1 => {
                // 64-bit `largesize` follows the 8-byte header (§4.2).
                let mut big = [0u8; 8];
                f.read_exact(&mut big)?;
                u64::from_be_bytes(big)
            }
            n => n,
        };
        if advance == 0 {
            return Err(std::io::Error::other("zero-size box while scanning for mdat"));
        }
        at += advance;
    }
    // `head` is `[0, at)` — ftyp + moov + any other pre-mdat boxes.
    let mut head = vec![0u8; at as usize];
    f.seek(SeekFrom::Start(0))?;
    f.read_exact(&mut head)?;
    Ok(head)
}

/// The AVI stream head: the leading `RIFF('AVI ' …)` up to and including the `movi` LIST
/// four-CC — everything `AviDemux::new` needs to discover streams at preroll. The `hdrl`
/// list (per-stream `strh`/`strf` headers) precedes `movi`, and `sc-avi`'s `probe_header`
/// resolves the tracks + the `movi` data start from these bytes; the whole file is fed on
/// the sink pad at run time regardless (like MKV/MP4).
///
/// 16 MiB is generous — real files put `hdrl` first, well under 1 MiB. Errors (rather than
/// panic) when no `movi` list appears in the bound — a truncated or non-AVI RIFF, which the
/// player reports cleanly.
#[allow(clippy::disallowed_methods)] // app-side header read before the pipeline; see module docs + clippy.toml
pub fn avi_head(path: &str) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let mut head = vec![0u8; 16 * 1024 * 1024];
    let n = f.read(&mut head)?;
    head.truncate(n);
    // The `movi` four-CC marks the start of the interleaved media chunks; the head is
    // everything up to and including it (`probe_header` wants the `movi` list header so it
    // can report the movi data start — the point the demuxer streams chunks from).
    let movi = head.windows(4).position(|w| w == b"movi").ok_or_else(|| {
        std::io::Error::other("no 'movi' list in the first 16 MiB — not a (supported) AVI file")
    })?;
    head.truncate(movi + 4);
    Ok(head)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A tiny, valid-enough MP4 skeleton: `ftyp`, then `moov`, then `mdat`. The head walk
    /// should stop at `mdat` and return exactly the `ftyp`+`moov` prefix.
    fn synth_mp4(ftyp_body: &[u8], moov_body: &[u8], mdat_body: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        let mut push_box = |kind: &[u8; 4], body: &[u8]| {
            let size = (8 + body.len()) as u32;
            v.extend_from_slice(&size.to_be_bytes());
            v.extend_from_slice(kind);
            v.extend_from_slice(body);
        };
        push_box(b"ftyp", ftyp_body);
        push_box(b"moov", moov_body);
        push_box(b"mdat", mdat_body);
        v
    }

    #[allow(clippy::disallowed_methods)] // test fixture write; clippy.toml sanctions test setup
    fn write_tmp(name: &str, bytes: &[u8]) -> String {
        let path = std::env::temp_dir().join(format!("scplay-head-test-{name}-{}", std::process::id()));
        let mut f = std::fs::File::create(&path).expect("create tmp");
        f.write_all(bytes).expect("write tmp");
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn mp4_head_stops_at_mdat() {
        let file = synth_mp4(b"isom\x00\x00\x02\x00mp41", b"MOOVDATA-metadata", b"\x00\x11\x22media");
        let path = write_tmp("faststart", &file);
        let head = mp4_head(&path).expect("faststart head walk");
        // The head is ftyp + moov, and does NOT include the mdat payload.
        assert_eq!(head.len(), 8 + b"isom\x00\x00\x02\x00mp41".len() + 8 + b"MOOVDATA-metadata".len());
        assert_eq!(&head[4..8], b"ftyp");
        assert!(!head.windows(4).any(|w| w == b"mdat"), "head must stop before mdat");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mp4_head_errors_without_mdat() {
        // ftyp + moov only, no mdat: the walk runs off the end and errors clearly (this is
        // also the shape a moov-after-mdat file degrades to relative to a forward walk).
        let mut file = Vec::new();
        for (k, b) in [(b"ftyp", &b"isom"[..]), (b"moov", &b"meta"[..])] {
            let size = (8 + b.len()) as u32;
            file.extend_from_slice(&size.to_be_bytes());
            file.extend_from_slice(k);
            file.extend_from_slice(b);
        }
        let path = write_tmp("nomdat", &file);
        let err = mp4_head(&path).expect_err("no mdat must error");
        assert!(err.to_string().contains("mdat"), "error should mention mdat: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mkv_header_errors_without_cluster() {
        // A file that begins with the EBML magic but never reaches a Cluster within the
        // (small, here) content: the scan must error, not scan forever or panic.
        let path = write_tmp("nocluster", &[0x1A, 0x45, 0xDF, 0xA3, 0x42, 0x00, 0x11]);
        let err = mkv_header_prefix(&path).expect_err("no cluster must error");
        assert!(err.to_string().contains("Cluster"), "error should mention Cluster: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn mkv_header_truncates_at_first_cluster() {
        // EBML magic + some metadata bytes + the Cluster id + media: the head must be
        // everything up to (not including) the Cluster id.
        let mut file = vec![0x1A, 0x45, 0xDF, 0xA3];
        file.extend_from_slice(b"segment-metadata-goes-here");
        let meta_len = file.len();
        file.extend_from_slice(id::CLUSTER); // the 4-byte Cluster id
        file.extend_from_slice(b"media-bytes-after");
        let path = write_tmp("cluster", &file);
        let head = mkv_header_prefix(&path).expect("head walk");
        assert_eq!(head.len(), meta_len, "head stops exactly at the Cluster id");
        let _ = std::fs::remove_file(&path);
    }
}
