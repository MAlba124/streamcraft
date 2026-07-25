//! Hardware VA-API decode tests. These *require a real device* and skip cleanly
//! (print + return) when [`sc_vaapi::probe`] reports none (or `SC_NO_VAAPI=1`), so
//! the suite is green on a machine without the hardware and exercises the real GPU
//! path where one exists.
//!
//! The end-to-end decode pulls the first access units of `../output.mkv` (a real
//! `V_MPEG4/ISO/AVC` file) through the element via `sc-mkv`'s reader + AVCC→Annex-B
//! reframer, and asserts plausible NV12 frames come out. Skipped when the fixture
//! is absent.

use std::path::Path;

use sc_mkv::{nal_head_from_config, MatroskaReader, Reframer};
use streamcraft_core::event::Event;
use streamcraft_core::format::Value;
use streamcraft_core::harness::Harness;
use streamcraft_core::time::Timestamp;

use sc_vaapi::VaapiH264Dec;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../output.mkv");
// The decoder frames can be large (e.g. 1920x1080 NV12 ≈ 3.1 MiB); size pool slots
// generously so the emit path is not the bottleneck of these tests.
const SLOT: usize = 8 * 1024 * 1024;

fn hw_available() -> bool {
    sc_vaapi::probe().is_some()
}

#[test]
fn probe_reports_h264() {
    let Some(caps) = sc_vaapi::probe() else {
        eprintln!("skip probe_reports_h264: no VA-API device");
        return;
    };
    eprintln!(
        "device={} va={}.{} vendor={:?} families={:?}",
        caps.device.display(),
        caps.version.0,
        caps.version.1,
        caps.vendor,
        caps.decode_families
    );
    // Every VA-API device we target for this element accelerates H.264 VLD.
    assert!(
        caps.supports("h264/annexb"),
        "expected h264/annexb in decode families, got {:?}",
        caps.decode_families
    );
}

/// Read the first `max_aus` H.264 Annex-B access units from `output.mkv`. Returns
/// `None` when the fixture is missing or has no AVC track. The parameter-set head
/// is prepended to the first access unit.
fn read_annexb_aus(max_aus: usize) -> Option<Vec<(Vec<u8>, bool)>> {
    if !Path::new(FIXTURE).exists() {
        return None;
    }
    let data = std::fs::read(FIXTURE).ok()?;
    let mut reader = MatroskaReader::new();
    // Feed in chunks until tracks are known, then keep feeding to gather frames.
    let mut fed = 0usize;
    let chunk = 1 << 20;
    // First: discover tracks.
    while !reader.tracks_ready() && fed < data.len() {
        let end = (fed + chunk).min(data.len());
        reader.push(&data[fed..end]).ok()?;
        fed = end;
    }
    let track = reader
        .tracks()
        .iter()
        .find(|t| t.codec_id == "V_MPEG4/ISO/AVC")?
        .clone();
    let (head, length_size) = nal_head_from_config(&track.codec_private, false).ok()?;
    let reframer = Reframer::Nal { length_size };

    let mut aus: Vec<(Vec<u8>, bool)> = Vec::new();
    let mut first = true;
    loop {
        while let Some(frame) = reader.next_frame() {
            if frame.track_number != track.track_number {
                continue;
            }
            let annexb = reframer.reframe_block(&frame.data).ok()?;
            let mut au = Vec::new();
            if first {
                au.extend_from_slice(&head);
                first = false;
            }
            au.extend_from_slice(&annexb);
            aus.push((au, frame.keyframe));
            if aus.len() >= max_aus {
                return Some(aus);
            }
        }
        if fed >= data.len() {
            break;
        }
        let end = (fed + chunk).min(data.len());
        reader.push(&data[fed..end]).ok()?;
        fed = end;
    }
    if aus.is_empty() {
        None
    } else {
        Some(aus)
    }
}

#[test]
fn decode_output_mkv_first_aus() {
    if !hw_available() {
        eprintln!("skip decode_output_mkv_first_aus: no VA-API device");
        return;
    }
    let Some(aus) = read_annexb_aus(30) else {
        eprintln!("skip decode_output_mkv_first_aus: fixture {FIXTURE} absent or no AVC track");
        return;
    };
    eprintln!("fed {} access units", aus.len());

    let mut h = Harness::with_slot_size(VaapiH264Dec::new(), SLOT);
    h.start().expect("start");

    let mut frames: Vec<streamcraft_core::buffer::Buffer> = Vec::new();
    for (i, (au, _kf)) in aus.iter().enumerate() {
        let mut buf = h.alloc(au);
        buf.pts = Timestamp::from_nanos(i as u64 * 40_000_000);
        h.push("sink", buf).expect("push");
        while let Some(f) = h.pull("src") {
            frames.push(f);
        }
    }
    // Flush the DPB.
    for f in h.eos().expect("eos") {
        frames.push(f);
    }

    // Report any warnings so a partial decode is visible.
    for m in h.bus_messages() {
        match m {
            streamcraft_core::bus::BusMessage::Warning { error, .. } => eprintln!("warn: {error:?}"),
            streamcraft_core::bus::BusMessage::Error { error, .. } => eprintln!("error: {error:?}"),
            _ => {}
        }
    }

    assert!(!frames.is_empty(), "expected at least one decoded frame");

    // Announced format: nv12 with plausible, positive dimensions.
    let announced = h.announced().expect("format announced");
    let vocab = h.vocabulary();
    let w_id = vocab.field_id("width").unwrap();
    let h_id = vocab.field_id("height").unwrap();
    let pf_id = vocab.field_id("pixfmt").unwrap();
    let width = match announced.get(w_id) {
        Some(Value::Int(w)) => w as u32,
        other => panic!("width not an Int: {other:?}"),
    };
    let height = match announced.get(h_id) {
        Some(Value::Int(hh)) => hh as u32,
        other => panic!("height not an Int: {other:?}"),
    };
    let pixfmt = match announced.get(pf_id) {
        Some(Value::Id(id)) => vocab.value_name(id).unwrap().to_string(),
        other => panic!("pixfmt not an Id: {other:?}"),
    };
    eprintln!("announced {width}x{height} {pixfmt}, {} frames", frames.len());
    assert_eq!(pixfmt, "nv12");
    assert!(width >= 16 && height >= 16, "implausible dims {width}x{height}");

    // Each NV12 frame is exactly w*h*3/2 tight bytes, and the Y plane is not a
    // constant (a real picture, not a cleared surface).
    let need = (width * height + width * height.div_ceil(2)) as usize;
    let f0 = &frames[0];
    assert_eq!(f0.memory.data().len(), need, "NV12 tight length mismatch");
    let y = &f0.memory.data()[..(width * height) as usize];
    let first = y[0];
    assert!(
        y.iter().any(|&b| b != first),
        "Y plane is constant — surface likely not decoded"
    );
}

#[test]
fn flush_then_continue() {
    if !hw_available() {
        eprintln!("skip flush_then_continue: no VA-API device");
        return;
    }
    let Some(aus) = read_annexb_aus(30) else {
        eprintln!("skip flush_then_continue: fixture absent");
        return;
    };

    let mut h = Harness::with_slot_size(VaapiH264Dec::new(), SLOT);
    h.start().expect("start");

    // Feed the first several AUs.
    let split = aus.len().min(10);
    for (au, _) in &aus[..split] {
        let buf = h.alloc(au);
        let _ = h.push("sink", buf);
        while h.pull("src").is_some() {}
    }

    // Flush (seek), then resume from the next keyframe onward.
    h.push_event(Event::FlushStart).expect("flush");
    // Find a keyframe at/after the split to resume cleanly (needs SPS/PPS: the
    // head-carrying first AU is a keyframe, so rebuild from a keyframe).
    let resume: Vec<_> = aus
        .iter()
        .enumerate()
        .filter(|(i, (_, kf))| *i >= split && *kf)
        .map(|(_, (au, _))| au.clone())
        .collect();
    let resume = if resume.is_empty() {
        // No later keyframe in the window — resume from the head-carrying first AU
        // (a keyframe) so the decoder re-primes SPS/PPS + an IDR.
        vec![aus[0].0.clone()]
    } else {
        resume
    };

    let mut got = 0;
    for au in &resume {
        let buf = h.alloc(au);
        let _ = h.push("sink", buf);
        while h.pull("src").is_some() {
            got += 1;
        }
    }
    for _ in h.eos().expect("eos") {
        got += 1;
    }
    for m in h.bus_messages() {
        match m {
            streamcraft_core::bus::BusMessage::Warning { error, .. } => eprintln!("warn: {error:?}"),
            streamcraft_core::bus::BusMessage::Error { error, .. } => eprintln!("error: {error:?}"),
            _ => {}
        }
    }
    assert!(got >= 1, "no frames produced after flush — VA context wedged");
}
