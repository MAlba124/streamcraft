//! `dump_track` — diagnostic: run a file through `MatroskaReader` + the demuxer's
//! reframing exactly as `MkvDemux` would, and write track 1's reframed byte stream
//! (codec head + frames) to a file for external inspection (`ffprobe`, hexdump,
//! or a reference decoder).
//!
//! ```text
//! cargo run --release -p sc-mkv --example dump_track -- IN.mkv OUT.bin [max_frames]
//! ```

use sc_mkv::codec::{family_for, nal_head_from_config, Reframer};
use sc_mkv::MatroskaReader;
use std::io::{Read, Write};

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(input), Some(output)) = (args.next(), args.next()) else {
        eprintln!("usage: dump_track IN.mkv OUT.bin [max_frames]");
        std::process::exit(2);
    };
    let max_frames: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    let mut reader = MatroskaReader::new();
    let mut file = std::fs::File::open(&input).expect("open input");
    let mut out = std::fs::File::create(&output).expect("create output");
    let mut chunk = vec![0u8; 256 * 1024];

    let mut reframer: Option<Reframer> = None;
    let mut wrote_head = false;
    let mut frames = 0usize;
    let mut dropped = 0usize;

    'outer: loop {
        let n = file.read(&mut chunk).expect("read");
        if n == 0 {
            break;
        }
        reader.push(&chunk[..n]).expect("parse");
        while let Some(frame) = reader.next_frame() {
            if frame.track_number != 1 {
                continue;
            }
            if reframer.is_none() {
                // Learn the codec + head from the discovered track, like MkvDemux does.
                let track = reader
                    .tracks()
                    .iter()
                    .find(|t| t.track_number == 1)
                    .expect("track 1 discovered")
                    .clone();
                let family = family_for(&track.codec_id);
                eprintln!(
                    "track 1: codec_id={} family={} codec_private={} bytes",
                    track.codec_id,
                    family,
                    track.codec_private.len()
                );
                let rf = match track.codec_id.as_str() {
                    "V_MPEG4/ISO/AVC" | "V_MPEGH/ISO/HEVC" => {
                        let is_hevc = track.codec_id == "V_MPEGH/ISO/HEVC";
                        match nal_head_from_config(&track.codec_private, is_hevc) {
                            Ok((head, length_size)) => {
                                eprintln!(
                                    "config record OK: head={} bytes, nal length_size={length_size}",
                                    head.len()
                                );
                                out.write_all(&head).unwrap();
                                wrote_head = true;
                                Reframer::Nal { length_size }
                            }
                            Err(e) => {
                                eprintln!("config record FAILED ({e:?}) — passthrough fallback");
                                Reframer::Passthrough
                            }
                        }
                    }
                    _ => Reframer::Passthrough,
                };
                reframer = Some(rf);
            }
            match reframer.as_ref().unwrap().reframe_block(&frame.data) {
                Ok(bytes) => {
                    out.write_all(&bytes).unwrap();
                    frames += 1;
                }
                Err(e) => {
                    dropped += 1;
                    if dropped <= 3 {
                        eprintln!("frame {} reframe FAILED: {e:?}", frames + dropped);
                    }
                }
            }
            if frames >= max_frames {
                break 'outer;
            }
        }
    }
    eprintln!("wrote {frames} frames ({dropped} dropped), head={wrote_head}");
}
