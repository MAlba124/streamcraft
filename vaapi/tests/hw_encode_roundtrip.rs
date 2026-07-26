//! Hardware VA-API **encode round-trip** tests: drive each encoder element with
//! deterministic raw frames, then decode its output with this workspace's
//! *independent, pure-Rust* decoders (sc-h264 / sc-h265 / sc-vp8) and gate on
//! PSNR against the source. A hardware encoder validated by a software decoder
//! that shares no code with it is the strongest hermetic check available — a
//! bitstream-header bug, a reference-management bug, or a chroma swap all
//! surface as decode failures or PSNR collapse.
//!
//! Like `hw_decode.rs` / `hw_encode.rs`, these need a real device and skip
//! cleanly (print + return) without one.
//!
//! Content: a moving diagonal gradient — smooth (so quality at the default QP is
//! high and the PSNR gate is far from the noise floor) but with real motion (so
//! P-frames actually predict).

use streamcraft_core::format::ValueDesc;
use streamcraft_core::harness::Harness;
use streamcraft_core::buffer::{Buffer, BufferFlags};
use streamcraft_core::time::Timestamp;

use sc_vaapi::h264parse;

const W: usize = 320;
const H: usize = 240;
const FRAMES: usize = 24;
/// Minimum acceptable mean PSNR (dB) across the round trip. Smooth content at
/// the default QP lands well above 35 dB; 30 leaves margin without letting a
/// real regression (wrong reference, wrong QP mapping, plane swap) through.
const MIN_PSNR_DB: f64 = 30.0;

/// One I420 frame of the moving gradient: luma is a diagonal triangle wave
/// sliding 4 px/frame, chroma a slow horizontal ramp with its own drift.
fn gradient_frame(t: usize) -> Vec<u8> {
    let (cw, ch) = (W / 2, H / 2);
    let mut v = Vec::with_capacity(W * H + 2 * cw * ch);
    let tri = |p: usize| -> u8 {
        let m = p % 510;
        if m < 255 { m as u8 } else { (510 - m) as u8 }
    };
    for y in 0..H {
        for x in 0..W {
            v.push(tri(x + y + 4 * t));
        }
    }
    for _y in 0..ch {
        for x in 0..cw {
            v.push(tri(2 * x + 3 * t + 60)); // Cb
        }
    }
    for _y in 0..ch {
        for x in 0..cw {
            v.push(tri(2 * x + 2 * t + 180)); // Cr
        }
    }
    v
}

/// Mean PSNR (dB) between two equal-length byte planes.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x as f64 - y as f64;
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Feed `FRAMES` gradient frames through an encoder harness; returns the encoded
/// buffers (flags intact).
fn encode_all(enc: &mut Harness) -> Vec<Buffer> {
    enc.fix_format(
        "sink",
        "video/raw",
        &[
            ("width", ValueDesc::Int(W as i64)),
            ("height", ValueDesc::Int(H as i64)),
            ("pixfmt", ValueDesc::Id("i420")),
            ("fps", ValueDesc::Rat(30, 1)),
        ],
    );
    let mut out = Vec::new();
    for t in 0..FRAMES {
        let mut buf = enc.alloc(&gradient_frame(t));
        buf.pts = Timestamp::from_millis((t as u64) * 33);
        enc.push("sink", buf).expect("encode push");
        while let Some(b) = enc.pull("src") {
            out.push(b);
        }
    }
    out.extend(enc.eos().expect("encoder eos"));
    out
}

/// Decode encoded buffers through a decoder harness; returns raw I420 frames.
fn decode_all(dec: &mut Harness, encoded: &[Buffer]) -> Vec<Vec<u8>> {
    let mut frames = Vec::new();
    for b in encoded {
        let mut copy = dec.alloc(b.memory.data());
        copy.pts = b.pts;
        copy.flags = b.flags;
        dec.push("sink", copy).expect("decode push");
        while let Some(f) = dec.pull("src") {
            frames.push(f.memory.data().to_vec());
        }
    }
    frames.extend(dec.eos().expect("decoder eos").into_iter().map(|f| f.memory.data().to_vec()));
    // Surface decode-side warnings — when a frame fails to decode, the reason
    // rides the bus, and a silent count mismatch is undebuggable.
    for m in dec.bus_messages() {
        use streamcraft_core::bus::BusMessage;
        match m {
            BusMessage::Warning { error, .. } => eprintln!("decoder warning: {error:?}"),
            BusMessage::Error { error, .. } => eprintln!("decoder error: {error:?}"),
            _ => {}
        }
    }
    frames
}

/// PSNR-gate `decoded` against the generated source frames.
fn assert_round_trip_quality(decoded: &[Vec<u8>], tag: &str) {
    assert_eq!(decoded.len(), FRAMES, "{tag}: all frames decode");
    let frame_len = W * H + 2 * (W / 2) * (H / 2);
    let mut total = 0.0;
    for (t, got) in decoded.iter().enumerate() {
        assert_eq!(got.len(), frame_len, "{tag}: frame {t} is tight I420");
        let src = gradient_frame(t);
        let p = psnr(&src, got);
        assert!(
            p > MIN_PSNR_DB - 8.0,
            "{tag}: frame {t} PSNR {p:.1} dB collapsed (mean gate is {MIN_PSNR_DB})"
        );
        total += p.min(100.0); // cap inf (identical frames) for a sane mean
    }
    let mean = total / decoded.len() as f64;
    eprintln!("{tag}: mean round-trip PSNR {mean:.1} dB over {FRAMES} frames");
    assert!(mean > MIN_PSNR_DB, "{tag}: mean PSNR {mean:.1} < {MIN_PSNR_DB}");
}

fn keyframe_flags_sane(encoded: &[Buffer], tag: &str) {
    assert!(!encoded.is_empty(), "{tag}: encoder produced output");
    assert!(
        encoded[0].flags.contains(BufferFlags::KEYFRAME),
        "{tag}: first frame is a keyframe"
    );
    // Default GOP (60) exceeds FRAMES — everything after the first is a delta.
    for (i, b) in encoded.iter().enumerate().skip(1) {
        assert!(b.flags.contains(BufferFlags::DELTA), "{tag}: frame {i} tagged DELTA");
    }
}

#[test]
fn h264_hw_encode_sw_decode_round_trip() {
    let Some(caps) = sc_vaapi::probe() else {
        eprintln!("skip h264 round trip: no VA-API device");
        return;
    };
    if !caps.supports_encode("h264/annexb") {
        eprintln!("skip h264 round trip: no H.264 encode entrypoint");
        return;
    }

    let mut enc = Harness::with_slot_size(sc_vaapi::VaapiH264Enc::new(), 512 * 1024);
    let encoded = encode_all(&mut enc);
    keyframe_flags_sane(&encoded, "h264");

    // Bitstream shape: the first access unit must open with Annex-B SPS + PPS +
    // IDR — the parameter sets this element authored.
    let first = encoded[0].memory.data();
    assert_eq!(&first[..4], &[0, 0, 0, 1], "h264: Annex-B start code");
    let types: Vec<u8> = h264parse::split_nals(first).iter().map(|n| n.unit_type).collect();
    assert!(types.contains(&h264parse::NAL_SPS), "h264: first AU has SPS, got {types:?}");
    assert!(types.contains(&h264parse::NAL_PPS), "h264: first AU has PPS, got {types:?}");
    assert!(
        types.contains(&h264parse::NAL_SLICE_IDR),
        "h264: first AU has an IDR slice, got {types:?}"
    );

    let mut dec = Harness::with_slot_size(sc_h264::H264Dec::new(), 512 * 1024);
    let decoded = decode_all(&mut dec, &encoded);
    assert_round_trip_quality(&decoded, "h264");
}

#[test]
fn h265_hw_encode_sw_decode_round_trip() {
    let Some(caps) = sc_vaapi::probe() else {
        eprintln!("skip h265 round trip: no VA-API device");
        return;
    };
    if !caps.supports_encode("h265/annexb") {
        eprintln!("skip h265 round trip: no HEVC encode entrypoint");
        return;
    }

    // The PSNR round trip runs **all-intra** (gop = 1): the reference decoder
    // (oxideav-h265) rejects the inter tooling of real-world encoders
    // (`InterNotSupported` — a documented sc-h265 caveat), so P frames cannot be
    // software-verified here regardless of encoder correctness. Inter output is
    // shape-checked below and validated externally (ffprobe/mpv, the muxer e2e).
    let mut enc = Harness::with_slot_size(sc_vaapi::VaapiH265Enc::new().with_gop(1), 512 * 1024);
    let encoded = encode_all(&mut enc);
    assert_eq!(encoded.len(), FRAMES, "h265: one AU per frame");
    for (i, b) in encoded.iter().enumerate() {
        assert!(
            b.flags.contains(BufferFlags::KEYFRAME),
            "h265: all-intra frame {i} tagged KEYFRAME"
        );
    }

    // First AU: VPS (32) + SPS (33) + PPS (34) + IDR_W_RADL (19). H.265 NAL type
    // lives in the high 6 bits of the first header byte (§7.3.1.2); the Annex-B
    // start-code framing is shared with H.264, so the splitter applies.
    let first = encoded[0].memory.data();
    assert_eq!(&first[..4], &[0, 0, 0, 1], "h265: Annex-B start code");
    let types: Vec<u8> =
        h264parse::split_nals(first).iter().map(|n| (n.raw[0] >> 1) & 0x3F).collect();
    for (want, name) in [(32u8, "VPS"), (33, "SPS"), (34, "PPS"), (19, "IDR_W_RADL")] {
        assert!(types.contains(&want), "h265: first AU has {name}, got {types:?}");
    }

    let mut dec = Harness::with_slot_size(sc_h265::H265Dec::new(), 512 * 1024);
    let decoded = decode_all(&mut dec, &encoded);
    assert_round_trip_quality(&decoded, "h265");

    // Default-GOP shape check: P frames come out as TRAIL_R (type 1) AUs.
    let mut enc = Harness::with_slot_size(sc_vaapi::VaapiH265Enc::new(), 512 * 1024);
    let encoded = encode_all(&mut enc);
    keyframe_flags_sane(&encoded, "h265 (default gop)");
    let p_types: Vec<u8> =
        h264parse::split_nals(encoded[1].memory.data()).iter().map(|n| (n.raw[0] >> 1) & 0x3F).collect();
    assert_eq!(p_types, vec![1], "h265: P access unit is a single TRAIL_R NAL");
}

/// The live congestion-control knobs: a mid-stream `force-keyframe` write must
/// produce an IDR on the very next frame, and a mid-stream `bitrate` retarget
/// must reach the driver's rate controller (frame sizes track the new target).
/// Noise frames force the encoder to actually spend bits, so the rate response
/// is measurable.
#[test]
fn h264_live_knobs_retarget_and_force_keyframe() {
    let Some(caps) = sc_vaapi::probe() else {
        eprintln!("skip live knobs: no VA-API device");
        return;
    };
    if !caps.supports_encode("h264/annexb") {
        eprintln!("skip live knobs: no H.264 encode entrypoint");
        return;
    }

    use streamcraft_core::event::Event;
    use streamcraft_core::format::Value;

    // Deterministic per-frame noise — incompressible, so VBR frame sizes are
    // bitrate-bound, not content-bound.
    let noise_frame = |t: usize| -> Vec<u8> {
        let len = W * H + 2 * (W / 2) * (H / 2);
        let mut v = Vec::with_capacity(len);
        let mut x = (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        for _ in 0..len {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            v.push((x >> 24) as u8);
        }
        v
    };

    // VBR at 4000 kbit/s with a 500 ms HRD window; gop long enough that no
    // scheduled IDR lands inside the test.
    let mut enc = Harness::with_slot_size(
        sc_vaapi::VaapiH264Enc::new().with_bitrate(4000).with_gop(200).with_hrd_ms(500),
        512 * 1024,
    );
    enc.fix_format(
        "sink",
        "video/raw",
        &[
            ("width", ValueDesc::Int(W as i64)),
            ("height", ValueDesc::Int(H as i64)),
            ("pixfmt", ValueDesc::Id("i420")),
            ("fps", ValueDesc::Rat(30, 1)),
        ],
    );

    let mut push_frame = |enc: &mut Harness, t: usize, out: &mut Vec<Buffer>| {
        let mut buf = enc.alloc(&noise_frame(t));
        buf.pts = Timestamp::from_millis((t as u64) * 33);
        enc.push("sink", buf).expect("encode push");
        while let Some(b) = enc.pull("src") {
            out.push(b);
        }
    };

    let mut out = Vec::new();
    for t in 0..30 {
        push_frame(&mut enc, t, &mut out);
    }
    // Live retarget: quarter the bitrate mid-stream.
    enc.push_event(Event::PropChanged { name: "bitrate", value: Value::Int(1000) })
        .expect("bitrate prop event");
    for t in 30..60 {
        push_frame(&mut enc, t, &mut out);
    }
    // Live forced keyframe.
    enc.push_event(Event::PropChanged { name: "force-keyframe", value: Value::Int(1) })
        .expect("force-keyframe prop event");
    for t in 60..64 {
        push_frame(&mut enc, t, &mut out);
    }
    out.extend(enc.eos().expect("encoder eos"));
    assert_eq!(out.len(), 64, "all frames encoded");

    // Rate response: skip 5 frames of BRC settling after the switch, then the
    // average frame size must clearly track the target drop. The full 4× is not
    // reachable on pure noise — the controller saturates at max QP (51) and
    // incompressible content has a hard floor there — so the gate is a robust
    // 1.5×: proof the retarget reached the hardware, not a rate-accuracy spec.
    let avg = |bufs: &[Buffer]| -> f64 {
        bufs.iter().map(|b| b.memory.data().len() as f64).sum::<f64>() / bufs.len() as f64
    };
    let before = avg(&out[10..30]);
    let after = avg(&out[35..60]);
    eprintln!("live bitrate: avg frame {before:.0} B @4000kbps → {after:.0} B @1000kbps");
    assert!(
        after < before / 1.5,
        "frame sizes must track the live bitrate drop (before {before:.0} B, after {after:.0} B)"
    );

    // Forced keyframe: frame 60 is an IDR (KEYFRAME flag + IDR NAL), its
    // neighbors are deltas (gop=200 schedules no IDR here).
    assert!(out[59].flags.contains(BufferFlags::DELTA), "frame 59 is a delta");
    assert!(out[60].flags.contains(BufferFlags::KEYFRAME), "frame 60 is the forced keyframe");
    assert!(out[61].flags.contains(BufferFlags::DELTA), "frame 61 is a delta");
    let types: Vec<u8> =
        h264parse::split_nals(out[60].memory.data()).iter().map(|n| n.unit_type).collect();
    assert!(
        types.contains(&h264parse::NAL_SLICE_IDR),
        "forced keyframe is a real IDR AU, got {types:?}"
    );

    // Decode integrity: the forced IDR must be a true random-access point — a
    // decoder joining at frame 60 (the teleconf late-joiner / loss-recovery
    // case) decodes everything from there. The *full* 64-frame stream is
    // ffmpeg-verified valid (64/64 frames, zero errors), but sc-h264's
    // underlying library mishandles GOPs longer than its inferred DPB window
    // (frames discarded un-output at the next IDR — a pre-existing decoder
    // caveat, like its h265 sibling's inter limitation), so the hermetic gate
    // here starts at the IDR.
    let mut dec = Harness::with_slot_size(sc_h264::H264Dec::new(), 512 * 1024);
    let decoded = decode_all(&mut dec, &out[60..]);
    assert_eq!(decoded.len(), 4, "the forced IDR opens a decodable stream from frame 60");
}

#[test]
fn vp8_hw_encode_sw_decode_round_trip() {
    let Some(caps) = sc_vaapi::probe() else {
        eprintln!("skip vp8 round trip: no VA-API device");
        return;
    };
    if !caps.supports_encode("vp8") {
        eprintln!("skip vp8 round trip: no VP8 encode entrypoint");
        return;
    }

    let mut enc = Harness::with_slot_size(sc_vaapi::VaapiVp8Enc::new(), 512 * 1024);
    let encoded = encode_all(&mut enc);
    keyframe_flags_sane(&encoded, "vp8");

    // Frame tag (RFC 6386 §9.1): bit 0 of the first byte is 0 for a key frame,
    // 1 for an interframe.
    assert_eq!(encoded[0].memory.data()[0] & 1, 0, "vp8: first frame tagged key");
    assert_eq!(encoded[1].memory.data()[0] & 1, 1, "vp8: second frame tagged inter");

    let mut dec = Harness::with_slot_size(sc_vp8::Vp8Dec::new(), 512 * 1024);
    let decoded = decode_all(&mut dec, &encoded);
    assert_round_trip_quality(&decoded, "vp8");
}
