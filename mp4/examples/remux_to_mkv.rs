//! Remux an MP4 into Matroska, untouched — **every passthrough-supported track**:
//! `filesrc ! mp4demux(passthrough) ⇉ mkvmuxn ! filesink`. No decode, no reframe — MP4's
//! length-prefixed NALs + the raw `avcC`/`hvcC` record are exactly Matroska's
//! `V_MPEG4/ISO/AVC` / `V_MPEGH/ISO/HEVC` shape, and MP4's raw AAC access units + the
//! esds AudioSpecificConfig are exactly `A_AAC` (RFC 9559 §12) — so every sample byte
//! survives verbatim and the whole run goes as fast as the disk allows (no clock —
//! transcoding never paces).
//!
//!     cargo run -p pf-mp4 --example remux_to_mkv -- input.mp4 [output.mkv]
//!
//! Tracks the muxer cannot carry (subtitles, unknown codecs, ASC-less mp4a) stay
//! unlinked and are dropped by scheduler policy — counted, not buffered.

use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use pf_mkv::MatroskaReader;
use pf_mp4::{Mp4Demux, Mp4Reader, Track};
use profluens_core::pipeline::Pipeline;
use profluens_elements::io::{FileSink, FileSrc};

/// Read the file head — everything up to (not including) `mdat` — by walking top-level
/// box headers with seeks, so a multi-GB file is never slurped whole. Requires a
/// `+faststart`-style layout (moov before mdat), which is also what streaming the file
/// through the demuxer requires.
fn read_head(path: &str) -> std::io::Result<Vec<u8>> {
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let mut at = 0u64;
    loop {
        if at + 8 > len {
            // No mdat found: not a progressive MP4 we can stream.
            return Err(std::io::Error::other("no mdat box (is this a faststart MP4?)"));
        }
        let mut hdr = [0u8; 8];
        f.seek(SeekFrom::Start(at))?;
        f.read_exact(&mut hdr)?;
        let size = u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as u64;
        if &hdr[4..8] == b"mdat" {
            break;
        }
        let advance = match size {
            0 => len - at, // box runs to EOF
            1 => {
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
    let mut head = vec![0u8; at as usize];
    f.seek(SeekFrom::Start(0))?;
    f.read_exact(&mut head)?;
    Ok(head)
}

/// Whether `mkvmuxn` can carry this track in passthrough mode, and roughly what as (for
/// the printout). Mirrors the demuxer's passthrough wiring: NAL tracks need their config
/// record (the CodecPrivate), `mp4a` needs the esds ASC (family promoted to `aac`),
/// `vp09` rides raw.
fn remuxable(t: &Track) -> Option<&'static str> {
    match &t.entry.kind {
        b"avc1" | b"avc3" if !t.entry.config_record.is_empty() => Some("h264/avcc"),
        b"hvc1" | b"hev1" if !t.entry.config_record.is_empty() => Some("h265/hvcc"),
        b"vp09" => Some("vp9"),
        _ if t.family() == "aac" => Some("aac"),
        _ => None,
    }
}

/// Bounded sanity pass over the produced MKV: parse the leading `budget` bytes, walk
/// every **video** block as length-prefixed NALs (length size from the record, ISO/IEC
/// 14496-15 §5.3.3.1), count audio blocks, and report. Catches the frame-split hazard (a
/// sample bigger than the demuxer's pool slot would mux as several broken blocks)
/// without re-reading the whole output.
fn verify_prefix(path: &str, budget: usize) -> std::io::Result<()> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; budget];
    let n = f.read(&mut buf)?;
    buf.truncate(n);

    let mut r = MatroskaReader::new();
    r.push(&buf).map_err(|e| std::io::Error::other(format!("output does not parse: {e:?}")))?;
    // The video track (the NAL walk target) + a summary of the rest.
    let (video_no, codec, private_len, length_size) = {
        let t = r
            .tracks()
            .iter()
            .find(|t| t.codec_id.starts_with("V_"))
            .ok_or_else(|| std::io::Error::other("no video track in output"))?;
        (
            t.track_number,
            t.codec_id.clone(),
            t.codec_private.len(),
            (t.codec_private.get(4).copied().unwrap_or(3) & 0x3) as usize + 1,
        )
    };
    let ntracks = r.tracks().len();

    let (mut frames, mut keyframes, mut audio_blocks) = (0u64, 0u64, 0u64);
    while let Some(fr) = r.next_frame() {
        if fr.track_number != video_no {
            audio_blocks += 1;
            continue;
        }
        // Every video block must walk cleanly as [len][NAL]…[len][NAL].
        let d = &fr.data;
        let mut off = 0usize;
        while off < d.len() {
            if off + length_size > d.len() {
                return Err(std::io::Error::other(format!(
                    "frame {frames}: truncated NAL length at {off} (split sample? \
                     raise the demuxer pool slot)"
                )));
            }
            let mut l = 0usize;
            for b in &d[off..off + length_size] {
                l = (l << 8) | *b as usize;
            }
            off += length_size + l;
        }
        if off != d.len() {
            return Err(std::io::Error::other(format!("frame {frames}: NAL walk overran the block")));
        }
        frames += 1;
        keyframes += fr.keyframe as u64;
    }
    println!(
        "verify: {ntracks} tracks, {codec} CodecPrivate {private_len}B, first {frames} \
         video blocks NAL-walk clean ({keyframes} keyframes), {audio_blocks} other-track \
         blocks interleaved"
    );
    Ok(())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(input) = args.next() else {
        eprintln!("usage: remux_to_mkv <input.mp4> [output.mkv]");
        std::process::exit(2);
    };
    let output = args
        .next()
        .unwrap_or_else(|| format!("{}.mkv", input.trim_end_matches(".mp4")));

    let head = read_head(&input).expect("read MP4 head (ftyp+moov)");
    println!("head: {} bytes (through moov)", head.len());

    // Pick every passthrough-remuxable track; the rest stay unlinked (dropped).
    let probe = Mp4Reader::new(&head).expect("resolve MP4 tables");
    let picked: Vec<(String, &'static str)> = probe
        .tracks()
        .iter()
        .filter_map(|t| {
            remuxable(t).map(|fam| {
                println!(
                    "track {}: {} → {fam}{}",
                    t.track_id,
                    String::from_utf8_lossy(&t.entry.kind),
                    if t.width != 0 { format!(" ({}x{})", t.width, t.height) } else { String::new() },
                );
                (format!("src_track{}", t.track_id), fam)
            })
        })
        .collect();
    assert!(!picked.is_empty(), "no remuxable track");
    println!("{} of {} tracks remuxed", picked.len(), probe.tracks().len());

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&input));
    let demux = p.add(Mp4Demux::passthrough(head));
    p.link((src, "src"), (demux, "sink")).expect("src -> demux");
    let added = p.preroll().expect("preroll");
    let mux = p.add(pf_mkv::MkvMux::multi(picked.len()));
    let sink = p.add(FileSink::new(&output));
    for (i, (pad_name, _)) in picked.iter().enumerate() {
        let ap = added
            .iter()
            .find(|ap| &ap.name == pad_name)
            .expect("picked pad discovered at preroll");
        p.link((ap.element, &ap.name), (mux, &format!("sink_{i}"))).expect("demux -> mux");
    }
    p.link((mux, "src"), (sink, "sink")).expect("mux -> sink");
    // Pool sizing under zero-copy retention (ZERO-COPY.md Stages 1+2):
    //
    // - The *filesrc* pool is the retained-chunk pool: the demuxer emits refcounted
    //   slices of the read buffers, and the muxer's open Cluster holds those slices —
    //   now for **every linked track** — until it closes, so the file-byte span of one
    //   Cluster (≤ 32.767 s of stream at the ms TimestampScale, typically one GOP; video
    //   + audio together) must fit in outstanding read slots, or the source stalls and
    //   the Cluster can never complete (a livelock, not just slowness). The muxer's
    //   pts-ordered merge also briefly queues whatever rides ahead of the other track's
    //   next frame — bounded by the file's own interleave skew, typically well under a
    //   second. 1 MiB × 96 ≈ 96 MiB still covers a 32 s A/V Cluster at ~20 Mbps.
    // - The *demuxer* pool only serves cold paths (in-band codec heads, chunk-straddle
    //   gathers): slots must still exceed the largest straddled sample so a straddle
    //   remuxes as one unsplit SimpleBlock (8 MiB × 8; allocated lazily, mostly unused).
    p.set_element_pool(src, 1 << 20, 96);
    p.set_element_pool(demux, 8 << 20, 8);

    let tap = p.tap_handle();
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let progress = {
        let done = std::sync::Arc::clone(&done);
        std::thread::spawn(move || {
            while !done.load(std::sync::atomic::Ordering::Acquire) {
                std::thread::sleep(std::time::Duration::from_millis(500));
                if let Some(s) = tap.snapshot(mux) {
                    eprint!("\rmuxed {:>8.1} MiB…", s.bytes_out as f64 / (1024.0 * 1024.0));
                }
            }
            eprintln!();
        })
    };

    let started = Instant::now();
    p.run().expect("remux run");
    done.store(true, std::sync::atomic::Ordering::Release);
    let _ = progress.join();
    let secs = started.elapsed().as_secs_f64();

    let out_len = std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0);
    println!(
        "wrote {output}: {:.1} MiB in {secs:.1}s ({:.0} MiB/s)",
        out_len as f64 / (1024.0 * 1024.0),
        out_len as f64 / (1024.0 * 1024.0) / secs.max(0.001),
    );
    verify_prefix(&output, 16 << 20).expect("verify remuxed output");
}
