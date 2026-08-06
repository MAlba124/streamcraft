//! `AudioConvert` in a pipeline: `filesrc(raw PCM) ! audioconvert ! filesink` rewrites the
//! interleaved PCM to a target [`SampleFormat`], byte-for-byte matching the pure conversion
//! library (spec: Formats — an `audioconvert`-class transform). The input format is pinned
//! with [`AudioConvert::with_input`], so no announcing upstream is needed — filesrc emits
//! raw `bytes` and the element interprets them via its construction-time input format.
//!
//! These also exercise the buffer-straddling carry: filesrc reads in pool-slot chunks that
//! split interchannel frames at arbitrary byte offsets, so the element must reassemble whole
//! frames before converting.
//!
//! The final test proves the *inference* path: [`AudioConvert::new`] (no `with_input`) learns
//! its whole input format — rate, channels and sample format — by name off the negotiated
//! `audio/raw` caps. A tiny in-test [`raw_src::RawAudioSrc`] advertises a concrete `audio/raw`
//! src, so link-time negotiation fixes those fields on the converter's sink and `new(target)`
//! reads them through the shared vocabulary with no out-of-band hint.

use profluens_audio::{
    convert_interleaved_vec, AudioConvert, AudioFormat, SampleFormat,
};
use profluens_core::pipeline::Pipeline;
use profluens_elements::io::{FileSink, FileSrc};

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf_audioconvert_{}_{}.bin", tag, std::process::id()));
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

/// A minimal test source with a *concrete* `audio/raw` src pad (fixed rate/channels/sample),
/// so link-time negotiation fixes those fields on a downstream sink and a consumer can infer
/// its input format from the negotiated caps — the piece `filesrc` (a `bytes` pad) can't
/// provide. It emits a fixed `S16` stereo @ 44100 PCM payload then EOS.
mod raw_src {
    use profluens_audio::{FAMILY, FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE};
    use profluens_core::batch::Inputs;
    use profluens_core::ctx::Ctx;
    use profluens_core::element::{
        Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
    };
    use profluens_core::error::Error;
    use profluens_core::event::Event;
    use profluens_core::format::{ConstraintDesc, FieldDesc, OfferDesc, ValueDesc};
    use profluens_core::id::PadId;
    use profluens_core::time::Timestamp;

    // Fixed `audio/raw`: S16 stereo @ 44100. `Eq` on every field so the intersection with a
    // broad `audio/raw` sink fixes all three (rate/channels as Int, sample as the interned
    // categorical id the converter reverses by name).
    pub const RATE: i64 = 44_100;
    pub const CHANNELS: i64 = 2;
    pub const SAMPLE: &str = "s16"; // == SampleFormat::S16.caps_name()

    static FIELDS: [FieldDesc; 3] = [
        FieldDesc { field: FIELD_RATE, allowed: ConstraintDesc::Eq(ValueDesc::Int(RATE)), preferred: None },
        FieldDesc { field: FIELD_CHANNELS, allowed: ConstraintDesc::Eq(ValueDesc::Int(CHANNELS)), preferred: None },
        FieldDesc { field: FIELD_SAMPLE, allowed: ConstraintDesc::Eq(ValueDesc::Id(SAMPLE)), preferred: None },
    ];
    static OFFERS: [OfferDesc; 1] = [OfferDesc { family: FAMILY, fields: &FIELDS }];
    static PADS: [PadDesc; 1] = [PadDesc {
        name: "src",
        direction: Direction::Src,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    }];
    static DESC: ElementDesc = ElementDesc {
        name: "rawaudiosrc",
        pads: &PADS,
        props: &[],
        sched: SchedHint::Active,
        inputs: InputPolicy::None,
        latency: LatencyDesc {
            min: Timestamp::ZERO,
            max: Timestamp::ZERO,
            is_live: false,
            jitter: Timestamp::ZERO,
        },
        make_default: None,
    };

    pub struct RawAudioSrc {
        pcm: Vec<u8>,
        off: usize,
    }

    impl RawAudioSrc {
        pub fn new(pcm: Vec<u8>) -> Self {
            Self { pcm, off: 0 }
        }
    }

    impl Element for RawAudioSrc {
        fn desc(&self) -> &'static ElementDesc {
            &DESC
        }
        fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
            self.off = 0;
            Ok(())
        }
        fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
            if self.off >= self.pcm.len() {
                return Ok(Flow::Eos);
            }
            let mut buf = match ctx.try_alloc(PadId(0)) {
                Some(b) => b,
                None => return Ok(Flow::Ok),
            };
            let cap = buf.memory.capacity();
            let n = cap.min(self.pcm.len() - self.off);
            buf.memory.as_mut_full()[..n].copy_from_slice(&self.pcm[self.off..self.off + n]);
            buf.memory.set_len(n);
            self.off += n;
            ctx.out(PadId(0)).push(buf);
            Ok(Flow::Ok)
        }
        fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
            Ok(())
        }
        fn stop(&mut self, _ctx: &mut Ctx) {}
    }
}

#[test]
fn new_infers_input_format_from_negotiated_caps() {
    // `rawaudiosrc(S16/2ch/44100) ! audioconvert(new(S32)) ! filesink`. The source fixes a
    // concrete `audio/raw` at link time, so `AudioConvert::new(S32)` — with NO `with_input` —
    // must infer the full input format (S16, 2ch, 44100) by name from `ctx.negotiated(sink)`
    // and produce byte-for-byte the same S32 as the pure library.
    let in_fmt = AudioFormat::new(
        raw_src::RATE as u32,
        raw_src::CHANNELS as u16,
        SampleFormat::S16,
    );
    // Enough frames that many pooled buffers flow and frames straddle buffer edges.
    let frames = 40_000usize;
    let mut pcm = Vec::with_capacity(frames * in_fmt.frame_stride());
    for i in 0..frames {
        let l = (i as i16).wrapping_mul(41).wrapping_add(3);
        let r = (i as i16).wrapping_mul(-29).wrapping_sub(9);
        pcm.extend_from_slice(&l.to_le_bytes());
        pcm.extend_from_slice(&r.to_le_bytes());
    }

    let outp = temp_path("infer_out");

    let mut p = Pipeline::new();
    let src = p.add(raw_src::RawAudioSrc::new(pcm.clone()));
    // The whole point: constructed WITHOUT the input format; it is inferred from caps.
    let conv = p.add(AudioConvert::new(SampleFormat::S32));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (conv, "sink")).expect("link rawaudiosrc->audioconvert");
    p.link((conv, "src"), (sink, "sink")).expect("link audioconvert->sink");
    p.run().expect("run");

    let got = std::fs::read(&outp).unwrap();
    let _ = std::fs::remove_file(&outp);

    let want = convert_interleaved_vec(SampleFormat::S16, SampleFormat::S32, 2, &pcm).unwrap();
    assert_eq!(got.len(), want.len(), "inferred-input conversion length (S16->S32, 2x wider)");
    assert_eq!(
        got, want,
        "AudioConvert::new inferred S16/2ch/44100 from negotiated caps and converted correctly"
    );
}
