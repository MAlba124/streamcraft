//! `hevc_decode` — on-device validation for `vaapih265dec`.
//!
//! Feeds a raw HEVC **Annex-B** elementary stream (extracted from one of the target
//! BluRay HEVC movies with `ffmpeg … -bsf:v hevc_mp4toannexb`) through the hardware
//! decode element and reports two things:
//!
//!   1. **PSNR per frame vs an ffmpeg I420 reference** — hardware decode should be
//!      essentially exact (very high PSNR; the only differences come from any
//!      NV12→I420 chroma repacking). A reference YUV of the same clip
//!      (`ffmpeg -i clip.hevc -f rawvideo -pix_fmt yuv420p ref.yuv`) is compared frame
//!      for frame.
//!   2. **Decode fps at 1080p** — the whole point of hardware decode is to hold
//!      realtime for the two 1080p movies the software `pf-h265` path cannot.
//!
//! Usage:
//! ```text
//! cargo run --release -p pf-vaapi --example hevc_decode -- <clip.hevc> [ref.yuv] [max_frames]
//! ```
//!
//! When no VA-API device (or no HEVC Main profile) is present, the element degrades
//! exactly like `vaapih264dec` (warn, disable) and this example reports that it could
//! not decode on-device — it still exercises the parse + submit code paths.

use std::time::Instant;

use profluens_core::harness::Harness;
use profluens_core::time::Timestamp;

use pf_vaapi::VaapiH265Dec;

// 1080p I420 ≈ 3.1 MiB; size pool slots generously so the emit path is never the
// bottleneck of the fps measurement.
const SLOT: usize = 8 * 1024 * 1024;

fn main() {
    let mut args = std::env::args().skip(1);
    let clip = args.next().unwrap_or_else(|| {
        eprintln!(
            "usage: hevc_decode <clip.hevc> [ref.yuv] [max_frames]\n\
             extract a clip with:\n  ffmpeg -i movie.mkv -map 0:v:0 -c copy \
             -bsf:v hevc_mp4toannexb -t 20 clip.hevc"
        );
        std::process::exit(2);
    });
    let ref_path = args.next();
    let max_frames: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(60);

    // Probe first so the "no device" case reports cleanly (and the fps number is
    // meaningful only on-device).
    match pf_vaapi::probe() {
        Some(caps) => {
            eprintln!(
                "device={} va={}.{} vendor={:?}",
                caps.device.display(),
                caps.version.0,
                caps.version.1,
                caps.vendor,
            );
            if !caps.supports("h265/annexb") {
                eprintln!(
                    "NOTE: this device does not advertise HEVC Main VLD decode \
                     (families={:?}); vaapih265dec will warn+disable. Compiling and \
                     the parse/submit path still run, but no frames will decode here.",
                    caps.decode_families
                );
            }
        }
        None => {
            eprintln!(
                "NOTE: no VA-API device (or PF_NO_VAAPI=1). vaapih265dec will \
                 warn+disable and emit no frames — nothing to decode on-device here."
            );
        }
    }

    let data = std::fs::read(&clip).unwrap_or_else(|e| {
        eprintln!("cannot read {clip}: {e}");
        std::process::exit(2);
    });
    let aus = split_access_units(&data);
    eprintln!("split {} access units from {} bytes of Annex-B ES", aus.len(), data.len());
    if aus.is_empty() {
        eprintln!("no access units found — is this a raw HEVC Annex-B stream?");
        std::process::exit(2);
    }

    // Optional reference YUV (I420 planar, same dimensions as the stream).
    let reference = ref_path.as_ref().map(|p| {
        std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("cannot read reference {p}: {e}");
            std::process::exit(2);
        })
    });

    let mut h = Harness::with_slot_size(VaapiH265Dec::new(), SLOT);
    h.start().expect("start");

    let mut frames: Vec<profluens_core::buffer::Buffer> = Vec::new();
    let start = Instant::now();
    for (i, au) in aus.iter().enumerate() {
        if frames.len() >= max_frames {
            break;
        }
        let mut buf = h.alloc(au);
        // ~24 fps nominal spacing; the element uses a feed-order pts FIFO.
        buf.pts = Timestamp::from_nanos(i as u64 * 41_708_333);
        let _ = h.push("sink", buf);
        while let Some(f) = h.pull("src") {
            frames.push(f);
        }
    }
    for f in h.eos().expect("eos") {
        frames.push(f);
    }
    let elapsed = start.elapsed();

    for m in h.bus_messages() {
        match m {
            profluens_core::bus::BusMessage::Warning { error, .. } => eprintln!("warn: {error:?}"),
            profluens_core::bus::BusMessage::Error { error, .. } => eprintln!("error: {error:?}"),
            _ => {}
        }
    }

    if frames.is_empty() {
        eprintln!(
            "\nno frames decoded — see the note above (no device / no HEVC profile), \
             or a bus warning explaining a dropped stream."
        );
        return;
    }

    // Report the announced format.
    let (mut width, mut height) = (0u32, 0u32);
    if let Some(announced) = h.announced() {
        let vocab = h.vocabulary();
        if let (Some(w_id), Some(h_id), Some(pf_id)) =
            (vocab.field_id("width"), vocab.field_id("height"), vocab.field_id("pixfmt"))
        {
            use profluens_core::format::Value;
            if let Some(Value::Int(w)) = announced.get(w_id) {
                width = w as u32;
            }
            if let Some(Value::Int(hh)) = announced.get(h_id) {
                height = hh as u32;
            }
            let pf = match announced.get(pf_id) {
                Some(Value::Id(id)) => vocab.value_name(id).unwrap_or("?").to_string(),
                _ => "?".into(),
            };
            eprintln!("announced {width}x{height} {pf}");
        }
    }

    // Decode fps: frames actually produced over wall-clock decode time.
    let secs = elapsed.as_secs_f64();
    let fps = if secs > 0.0 { frames.len() as f64 / secs } else { f64::INFINITY };
    println!("decoded {} frames in {:.3}s → {:.1} fps", frames.len(), secs, fps);
    if width >= 1280 {
        println!(
            "  ({}x{}: {} realtime for 24 fps playback)",
            width,
            height,
            if fps >= 24.0 { "holds" } else { "BELOW" }
        );
    }

    // PSNR vs the ffmpeg I420 reference, frame for frame.
    if let (Some(ref_data), true) = (&reference, width > 0 && height > 0) {
        let frame_len = (width * height + 2 * (width.div_ceil(2) * height.div_ceil(2))) as usize;
        let n = (ref_data.len() / frame_len).min(frames.len());
        if n == 0 {
            eprintln!(
                "reference too small or dimension mismatch (frame_len={frame_len}, ref={} bytes)",
                ref_data.len()
            );
        } else {
            let mut sum = 0.0;
            let mut worst = f64::INFINITY;
            for i in 0..n {
                let dec = frames[i].memory.data();
                let rf = &ref_data[i * frame_len..(i + 1) * frame_len];
                if dec.len() < frame_len {
                    continue;
                }
                let p = psnr(&dec[..frame_len], rf);
                sum += p;
                worst = worst.min(p);
                // Print the first few, and any frame that falls below the exact floor
                // (a reference/reorder divergence shows up as a localized dip; a clean
                // hardware decode is byte-exact → `inf`). Split Y vs chroma so a
                // chroma-only regression is legible.
                if i < 8 || p < 45.0 {
                    let y = (width * height) as usize;
                    let py = psnr(&dec[..y], &rf[..y]);
                    let pc = psnr(&dec[y..frame_len], &rf[y..frame_len]);
                    println!("  frame {i:2}: PSNR = {p:.2} dB (Y {py:.2}, C {pc:.2})");
                }
            }
            println!(
                "PSNR over {n} frames: mean {:.2} dB, worst {:.2} dB {}",
                sum / n as f64,
                worst,
                if worst > 45.0 { "(hardware-exact)" } else { "(!! below 45 dB floor)" }
            );
        }
    } else if reference.is_none() {
        eprintln!("(no reference YUV given — pass ref.yuv as the 2nd arg for PSNR)");
    }
}

/// Peak-signal-to-noise ratio (dB) of two equal-length 8-bit byte planes. `inf` when
/// identical (exact hardware decode).
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut mse = 0.0f64;
    for i in 0..n {
        let d = a[i] as f64 - b[i] as f64;
        mse += d * d;
    }
    mse /= n as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Split a raw HEVC Annex-B elementary stream into access units — one coded picture
/// per AU, with its preceding parameter-set / SEI / AUD NALs (H.265 §7.4.2.4.4). A
/// new AU begins at a VCL NAL whose `first_slice_segment_in_pic_flag == 1` when the
/// current AU already holds a VCL NAL, or at a VPS/SPS/PPS/AUD/prefix-SEI following
/// VCL data. Total (no panic) on arbitrary input.
fn split_access_units(es: &[u8]) -> Vec<Vec<u8>> {
    // Collect start-code positions (3- or 4-byte).
    let mut starts: Vec<usize> = Vec::new();
    let mut i = 0;
    while i + 3 <= es.len() {
        if es[i] == 0 && es[i + 1] == 0 && es[i + 2] == 1 {
            starts.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut aus: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut cur_has_vcl = false;

    for (k, &s) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(es.len());
        // Payload begins after the start code (3-byte here; a 4-byte code is a 3-byte
        // one with a leading extra 0 that belongs to the previous NAL's trailer).
        let payload = s + 3;
        if payload + 1 >= end {
            continue;
        }
        let nal_type = (es[payload] >> 1) & 0x3f;
        let is_vcl = nal_type <= 31;
        let first_slice = is_vcl && (es[payload + 2] & 0x80) != 0; // first_slice_segment_in_pic_flag

        // Boundary: a new picture's first slice, or a param/AUD after VCL data.
        let new_au = if is_vcl {
            first_slice && cur_has_vcl
        } else {
            cur_has_vcl && matches!(nal_type, 32 | 33 | 34 | 35 | 39)
        };
        if new_au && !cur.is_empty() {
            aus.push(std::mem::take(&mut cur));
            cur_has_vcl = false;
        }
        cur.extend_from_slice(&es[s..end]);
        if is_vcl {
            cur_has_vcl = true;
        }
    }
    if !cur.is_empty() {
        aus.push(cur);
    }
    aus
}
