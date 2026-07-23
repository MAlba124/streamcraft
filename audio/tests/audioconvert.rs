//! `AudioConvert` in a pipeline: `filesrc(raw PCM) ! audioconvert ! filesink` rewrites the
//! interleaved PCM to a target [`SampleFormat`], byte-for-byte matching the pure conversion
//! library (spec: Formats — an `audioconvert`-class transform). The input format is pinned
//! with [`AudioConvert::with_input`], so no announcing upstream is needed — filesrc emits
//! raw `bytes` and the element interprets them via its construction-time input format.
//!
//! These also exercise the buffer-straddling carry: filesrc reads in pool-slot chunks that
//! split interchannel frames at arbitrary byte offsets, so the element must reassemble whole
//! frames before converting.

use streamcraft_audio::{
    convert_interleaved_vec, AudioConvert, AudioFormat, SampleFormat,
};
use streamcraft_core::pipeline::Pipeline;
use streamcraft_elements::io::{FileSink, FileSrc};

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("sc_audioconvert_{}_{}.bin", tag, std::process::id()));
    p
}

/// Run `filesrc(input_pcm) ! AudioConvert::with_input(in_fmt, target) ! filesink` and return
/// the bytes the sink wrote.
fn run_convert(input_pcm: &[u8], in_fmt: AudioFormat, target: SampleFormat, tag: &str) -> Vec<u8> {
    let inp = temp_path(&format!("{tag}_in"));
    let outp = temp_path(&format!("{tag}_out"));
    std::fs::write(&inp, input_pcm).unwrap();

    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&inp));
    let conv = p.add(AudioConvert::with_input(in_fmt, target));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (conv, "sink")).expect("link src->audioconvert");
    p.link((conv, "src"), (sink, "sink")).expect("link audioconvert->sink");
    p.run().expect("run");

    let got = std::fs::read(&outp).unwrap();
    let _ = std::fs::remove_file(&inp);
    let _ = std::fs::remove_file(&outp);
    got
}

#[test]
fn s16_stereo_to_s32_matches_library_over_a_large_stream() {
    // > 256 KiB of stereo S16 so many pooled buffers flow and frames straddle buffer edges.
    let in_fmt = AudioFormat::new(48_000, 2, SampleFormat::S16);
    let frames = 80_000usize; // stereo s16 -> 320_000 payload bytes
    let mut pcm = Vec::with_capacity(frames * 4);
    for i in 0..frames {
        let l = (i as i16).wrapping_mul(37).wrapping_add(11);
        let r = (i as i16).wrapping_mul(-53).wrapping_sub(7);
        pcm.extend_from_slice(&l.to_le_bytes());
        pcm.extend_from_slice(&r.to_le_bytes());
    }

    let got = run_convert(&pcm, in_fmt, SampleFormat::S32, "s16_s32");

    // Ground truth from the pure library (independently unit-tested against known values).
    let want = convert_interleaved_vec(SampleFormat::S16, SampleFormat::S32, 2, &pcm).unwrap();
    assert_eq!(got.len(), want.len(), "converted length (s16->s32, 4x wider payload)");
    assert_eq!(got, want, "element output matches the conversion library byte-for-byte");
}

#[test]
fn s16_to_s32_exact_known_bytes() {
    // Ground-truth independent of the library: s16 -> s32 is `<< 16`, so each 16-bit sample
    // becomes that value shifted left 16 bits, little-endian. Hand-verify a few samples.
    let in_fmt = AudioFormat::new(44_100, 1, SampleFormat::S16);
    let samples: [i16; 4] = [0, 1, -1, i16::MAX];
    let mut pcm = Vec::new();
    for s in samples {
        pcm.extend_from_slice(&s.to_le_bytes());
    }

    let got = run_convert(&pcm, in_fmt, SampleFormat::S32, "known");

    let mut want = Vec::new();
    for s in samples {
        want.extend_from_slice(&((s as i32) << 16).to_le_bytes());
    }
    assert_eq!(got, want, "s16->s32 produces exactly (s << 16) little-endian per sample");
}

#[test]
fn f32_to_s16_full_scale_and_clamp() {
    // Float -> int with clamping, end to end. +1.0 -> 32767, -1.0 -> -32767, and out-of-range
    // clamps rather than wraps.
    let in_fmt = AudioFormat::new(48_000, 1, SampleFormat::F32);
    let samples: [f32; 5] = [0.0, 1.0, -1.0, 2.0, -9.0];
    let mut pcm = Vec::new();
    for s in samples {
        pcm.extend_from_slice(&s.to_le_bytes());
    }

    let got = run_convert(&pcm, in_fmt, SampleFormat::S16, "f32_s16");

    let want: Vec<u8> = [0i16, 32767, -32767, 32767, -32767]
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    assert_eq!(got, want, "f32->s16 scales by (2^15 - 1) and clamps at full scale");
}

#[test]
fn u8_to_s16_carries_the_128_bias() {
    // U8 is unsigned-biased-by-128. Byte 128 is silence (0), 0 is full-scale negative, 255 is
    // near full-scale positive. Widening is (u - 128) << 8.
    let in_fmt = AudioFormat::new(8_000, 2, SampleFormat::U8);
    let bytes: [u8; 8] = [128, 0, 255, 129, 127, 200, 56, 128];
    let got = run_convert(&bytes, in_fmt, SampleFormat::S16, "u8_s16");

    let want: Vec<u8> = bytes
        .iter()
        .flat_map(|&u| ((((u as i32) - 128) << 8) as i16).to_le_bytes())
        .collect();
    assert_eq!(got, want, "u8->s16 applies the 128 bias then widens by << 8");
}

#[test]
fn passthrough_when_target_equals_input() {
    // Target == input sample format: the payload must pass through byte-identically.
    let in_fmt = AudioFormat::new(44_100, 2, SampleFormat::S24);
    let frames = 5_000usize; // stereo s24 -> 30_000 bytes
    let mut pcm = Vec::with_capacity(frames * 6);
    for i in 0..frames {
        // Two distinct 24-bit-ish values per frame.
        let l = (i as i32 * 7) & 0x00FF_FFFF;
        let r = (i as i32 * -3) & 0x00FF_FFFF;
        pcm.extend_from_slice(&l.to_le_bytes()[..3]);
        pcm.extend_from_slice(&r.to_le_bytes()[..3]);
    }

    let got = run_convert(&pcm, in_fmt, SampleFormat::S24, "passthrough");
    assert_eq!(got, pcm, "identical sample format passes the PCM through unchanged");
}

#[test]
fn s32_to_s16_narrows_with_rounding_matches_library() {
    // Narrowing (with rounding + clamping) end to end, cross-checked against the library.
    let in_fmt = AudioFormat::new(48_000, 2, SampleFormat::S32);
    let frames = 20_000usize;
    let mut pcm = Vec::with_capacity(frames * 8);
    for i in 0..frames {
        // Values with non-zero low 16 bits so rounding actually engages, plus the extremes.
        let l = (i as i32).wrapping_mul(0x0001_3579).wrapping_add(0x0000_8000);
        let r = i32::MAX.wrapping_sub(i as i32);
        pcm.extend_from_slice(&l.to_le_bytes());
        pcm.extend_from_slice(&r.to_le_bytes());
    }

    let got = run_convert(&pcm, in_fmt, SampleFormat::S16, "s32_s16");
    let want = convert_interleaved_vec(SampleFormat::S32, SampleFormat::S16, 2, &pcm).unwrap();
    assert_eq!(got.len(), want.len(), "narrowed payload is half the width");
    assert_eq!(got, want, "s32->s16 narrowing (round+clamp) matches the library");
}
