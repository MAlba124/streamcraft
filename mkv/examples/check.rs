//! Full-file MKV structural check (debug tool): stream the file through
//! [`MatroskaReader`], NAL-walk every block as length-prefixed NALs (for `V_MPEG4/…`
//! tracks), and report timestamps/keyframes/anomalies — the whole file, bounded
//! memory, unlike a player's opaque "corrupt file detected".
//!
//!     cargo run -p pf-mkv --example check -- file.mkv

use std::io::Read;

use pf_mkv::MatroskaReader;

fn main() {
    let path = std::env::args().nth(1).expect("usage: check <file.mkv>");
    let mut f = std::fs::File::open(&path).expect("open");

    let mut r = MatroskaReader::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut fed = 0u64;
    let mut frames = 0u64;
    let mut keyframes = 0u64;
    let mut nal_bad = 0u64;
    let mut last_pts = 0u64;
    let mut regressions = 0u64;
    let mut max_regress = 0i64;
    let mut length_size = 4usize;
    let mut checked_private = false;

    loop {
        let n = f.read(&mut buf).expect("read");
        if n == 0 {
            break;
        }
        if let Err(e) = r.push(&buf[..n]) {
            println!("PARSE ERROR after feeding {} bytes: {e:?}", fed + n as u64);
            break;
        }
        fed += n as u64;
        if !checked_private {
            if let Some(t) = r.tracks().first() {
                length_size = (t.codec_private.get(4).copied().unwrap_or(3) & 0x3) as usize + 1;
                println!(
                    "track: {} {}x{}, CodecPrivate {}B, NAL length size {}",
                    t.codec_id, t.pixel_width, t.pixel_height, t.codec_private.len(), length_size
                );
                checked_private = true;
            }
        }
        while let Some(fr) = r.next_frame() {
            // NAL walk.
            let d = &fr.data;
            let mut off = 0usize;
            let mut ok = true;
            while off < d.len() {
                if off + length_size > d.len() {
                    ok = false;
                    break;
                }
                let mut l = 0usize;
                for b in &d[off..off + length_size] {
                    l = (l << 8) | *b as usize;
                }
                off += length_size + l;
            }
            if !ok || off != d.len() {
                nal_bad += 1;
                if nal_bad <= 5 {
                    println!("frame {frames}: bad NAL walk (len {} pts {}ms)", d.len(), fr.pts_ns / 1_000_000);
                }
            }
            if fr.pts_ns < last_pts {
                regressions += 1;
                max_regress = max_regress.max((last_pts - fr.pts_ns) as i64);
            }
            last_pts = fr.pts_ns;
            frames += 1;
            keyframes += fr.keyframe as u64;
        }
    }
    println!(
        "fed {fed} bytes: {frames} frames, {keyframes} keyframes, {nal_bad} bad NAL walks, \
         {regressions} pts regressions (max {} ms), last pts {} ms",
        max_regress / 1_000_000,
        last_pts / 1_000_000
    );
}
