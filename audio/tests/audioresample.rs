//! `AudioResample` in a pipeline: `rawaudiosrc ! audioresample ! filesink` band-limits and
//! rate-converts interleaved PCM (spec: Formats — sample-rate conversion). Two things are proven
//! end to end that the standalone DSP unit tests (in `resample.rs`) cannot:
//!
//! 1. **Inference + announce**: [`AudioResample::new`] (no `with_input`) learns its input format
//!    by name from the negotiated `audio/raw` caps a source fixes at link time, and announces its
//!    output rate downstream — the full dynamic-caps consumer+producer path.
//! 2. **The element wiring**: decode → de-interleave → per-channel streaming resample →
//!    re-interleave → re-encode, across pool-sized buffers with the partial-frame carry.
//!
//! The signal is a pure 1 kHz sine at 44100; after resampling to 48000 the recovered PCM must
//! still be a 1 kHz sine (dominant frequency preserved) with the ratio-correct sample count.

use profluens_audio::{output_len, AudioFormat, AudioResample, SampleFormat};
use profluens_core::pipeline::Pipeline;
use profluens_elements::io::FileSink;

const PI: f64 = std::f64::consts::PI;

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf_audioresample_{}_{}.bin", tag, std::process::id()));
    p
}

/// A minimal source with a concrete `audio/raw` src pad (fixed rate/channels/sample), so
/// link-time negotiation fixes those on the resampler's sink and `AudioResample::new` can infer
/// its input format — the piece `filesrc` (a `bytes` pad) can't provide. Emits S16 mono @ 44100.
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

    pub const RATE: i64 = 44_100;
    pub const CHANNELS: i64 = 1;
    pub const SAMPLE: &str = "s16";

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

/// Single-bin DFT magnitude at `f` Hz over S16 samples read from `pcm` (mono), sampled at `rate`.
fn mag_at_s16(pcm: &[u8], rate: u32, f: f64) -> f64 {
    let (mut re, mut im) = (0.0f64, 0.0f64);
    let w = 2.0 * PI * f / rate as f64;
    let n = pcm.len() / 2;
    for i in 0..n {
        let s = i16::from_le_bytes([pcm[i * 2], pcm[i * 2 + 1]]) as f64 / 32768.0;
        let ph = w * i as f64;
        re += s * ph.cos();
        im -= s * ph.sin();
    }
    (re * re + im * im).sqrt() / n as f64
}

/// Coarse dominant-frequency search over `[f_lo, f_hi]` at `step` Hz.
fn dominant_s16(pcm: &[u8], rate: u32, f_lo: f64, f_hi: f64, step: f64) -> (f64, f64) {
    let mut best = (f_lo, -1.0);
    let mut f = f_lo;
    while f <= f_hi {
        let m = mag_at_s16(pcm, rate, f);
        if m > best.1 {
            best = (f, m);
        }
        f += step;
    }
    best
}

#[test]
fn new_infers_input_and_resamples_sine_44100_to_48000() {
    // `rawaudiosrc(S16 mono 44100, 1 kHz sine) ! audioresample(new(48000)) ! filesink`.
    // AudioResample::new — NO with_input — must infer 44100/mono/S16 from negotiated caps,
    // announce 48000 output, and produce a 1 kHz sine at 48000 with the ratio-correct length.
    let in_fmt = AudioFormat::new(raw_src::RATE as u32, raw_src::CHANNELS as u16, SampleFormat::S16);
    let target = 48_000u32;
    let freq = 1_000.0;

    let frames = 44_100usize; // ~1 s
    let mut pcm = Vec::with_capacity(frames * in_fmt.frame_stride());
    for n in 0..frames {
        let s = (0.7 * (2.0 * PI * freq * n as f64 / in_fmt.sample_rate as f64).sin()) as f32;
        let v = (s * 32767.0).round().clamp(-32768.0, 32767.0) as i16;
        pcm.extend_from_slice(&v.to_le_bytes());
    }

    let outp = temp_path("sine_out");
    let mut p = Pipeline::new();
    let src = p.add(raw_src::RawAudioSrc::new(pcm));
    let res = p.add(AudioResample::new(target)); // input inferred from caps
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (res, "sink")).expect("link rawaudiosrc->audioresample");
    p.link((res, "src"), (sink, "sink")).expect("link audioresample->sink");
    p.run().expect("run");

    let got = std::fs::read(&outp).unwrap();
    let _ = std::fs::remove_file(&outp);

    // Output length ≈ input · 48000/44100 (within the FIR's fixed start-up slop).
    let out_frames = (got.len() / 2) as i64;
    let want_frames = output_len(frames as u64, in_fmt.sample_rate, target) as i64;
    assert!(
        (out_frames - want_frames).abs() <= 4,
        "resampled length: got {out_frames} frames, want ~{want_frames}"
    );

    // Analyse the steady-state body (skip the FIR transient at both ends).
    let trim = 1_000usize * 2; // bytes (S16 mono)
    assert!(got.len() > 3 * trim, "output too short: {} bytes", got.len());
    let core = &got[trim..got.len() - trim];

    // The dominant frequency near 1 kHz is preserved.
    let (peak, peak_mag) = dominant_s16(core, target, 500.0, 1_500.0, 2.0);
    assert!((peak - freq).abs() <= 3.0, "dominant freq {peak} Hz, want {freq}");
    // Amplitude 0.7 sine → single-bin magnitude ≈ 0.35; allow a generous band.
    assert!(peak_mag > 0.25 && peak_mag < 0.45, "peak magnitude {peak_mag} (gain error?)");
    // No large aliasing image elsewhere in the band.
    let alias = mag_at_s16(core, target, 5_000.0);
    assert!(alias < peak_mag * 0.05, "alias image {alias} too large vs peak {peak_mag}");
}
