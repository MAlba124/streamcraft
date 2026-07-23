//! Incremental FLAC decoding + the dynamic-caps proof.
//!
//! 1. [`StreamDecoder`] fed in awkward small chunks recovers exactly what the proven
//!    whole-buffer [`FlacDecoder`] does — proving it decodes frame-by-frame without
//!    buffering the whole stream (spec: latency #2).
//! 2. `filesrc ! flacdec ! probe`: [`FlacDec`] announces its `audio/raw` format at runtime
//!    (from STREAMINFO), which reaches the downstream probe as an
//!    [`Event::FormatChange`] — proving runtime caps propagation end to end (spec:
//!    Formats — dynamic caps).

use std::sync::{Arc, Mutex};

use sc_flac::{FlacDec, FlacDecoder, FlacEncoder, SampleFormat, StreamDecoder};
use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::{FixedFormat, OfferDesc, Value};
use streamcraft_core::pipeline::Pipeline;
use streamcraft_core::time::Timestamp;
use streamcraft_elements::io::FileSrc;

/// Encode interleaved S16 samples into a complete FLAC stream (mirrors `roundtrip.rs`:
/// encode frames, then patch the header with the finalised STREAMINFO body).
fn encode_s16(samples: &[i16], channels: u32, rate: u32) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        pcm.extend_from_slice(&s.to_le_bytes());
    }
    let (mut enc, mut header) = FlacEncoder::new(rate, channels, SampleFormat::S16).expect("enc");
    let mut frames = Vec::new();
    enc.encode_interleaved(&pcm, &mut frames).expect("encode");
    let body = enc.finish();
    header[sc_flac::streaminfo_offset()..sc_flac::streaminfo_offset() + body.len()]
        .copy_from_slice(&body);
    let mut stream = header;
    stream.extend_from_slice(&frames);
    stream
}

/// A couple of detuned sines per channel — non-trivial, compressible content.
fn signal(frames: usize, channels: u32) -> Vec<i16> {
    let mut v = Vec::with_capacity(frames * channels as usize);
    for i in 0..frames {
        for c in 0..channels {
            let phase = 2.0 * std::f64::consts::PI * (3.0 + c as f64) * i as f64 / 200.0;
            v.push((9000.0 * phase.sin()) as i16);
        }
    }
    v
}

/// Decode `stream` by pushing it in `chunk`-sized pieces, returning all samples.
fn decode_chunked(stream: &[u8], chunk: usize) -> Vec<i64> {
    let mut sd = StreamDecoder::new();
    let mut got: Vec<i64> = Vec::new();
    for part in stream.chunks(chunk) {
        sd.push(part);
        while let Some(frame) = sd.pull().expect("incremental decode") {
            got.extend_from_slice(&frame.samples);
        }
    }
    got
}

#[test]
fn stream_decoder_is_incremental_and_lossless() {
    // Multi-frame streams across a range of chunk sizes.
    for &(frames, ch) in &[(9_001usize, 2u32), (5_000, 1), (1, 2)] {
        let samples = signal(frames, ch);
        let expected: Vec<i64> = samples.iter().map(|&s| s as i64).collect();
        let stream = encode_s16(&samples, ch, 44100);

        // The proven whole-buffer decoder is the reference.
        let reference = FlacDecoder::decode(&stream).expect("decode").samples;
        assert_eq!(reference, expected, "whole-buffer decode lossless (frames={frames} ch={ch})");

        for &chunk in &[13usize, 137, 4096] {
            assert_eq!(
                decode_chunked(&stream, chunk),
                expected,
                "incremental (chunk={chunk}) lossless (frames={frames} ch={ch})"
            );
        }
    }

    // The hardest incremental case — one byte per push, so a frame straddles hundreds of
    // pushes — on a small stream (kept small because re-attempting the frame per byte is
    // quadratic; correctness is the point, not throughput).
    let samples = signal(600, 2);
    let expected: Vec<i64> = samples.iter().map(|&s| s as i64).collect();
    let stream = encode_s16(&samples, 2, 44100);
    assert_eq!(decode_chunked(&stream, 1), expected, "byte-at-a-time incremental lossless");
}

// --- Dynamic-caps proof -------------------------------------------------------------

static PROBE_OFFERS: [OfferDesc; 1] = [OfferDesc::any("audio/raw")];
static PROBE_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &PROBE_OFFERS,
    dynamic: false,
    validate: None,
}];
static PROBE_DESC: ElementDesc = ElementDesc {
    name: "formatprobe",
    pads: &PROBE_PADS,
    props: &[],
    sched: SchedHint::Active, // its own group, so the FormatChange crosses the ring
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// A sink that records the `audio/raw` format it is told about at runtime.
struct FormatProbe {
    seen: Arc<Mutex<Option<FixedFormat>>>,
}

impl Element for FormatProbe {
    fn desc(&self) -> &'static ElementDesc {
        &PROBE_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        while inputs.pop().is_some() {} // consume the decoded PCM
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // The decoder's runtime announcement arrives here as a FormatChange (spec:
        // dynamic caps) — record the concrete format it carries.
        if let Event::FormatChange(f) = event {
            *self.seen.lock().unwrap() = Some(f.clone());
        }
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("sc_flacdec_{}_{}.flac", tag, std::process::id()));
    p
}

#[test]
fn flacdec_announces_runtime_audio_format() {
    let samples = signal(10_000, 2);
    let stream = encode_s16(&samples, 2, 44100);
    let path = temp_path("probe");
    std::fs::write(&path, &stream).unwrap();

    let seen = Arc::new(Mutex::new(None));
    let mut p = Pipeline::new();
    let src = p.add(FileSrc::new(&path));
    let dec = p.add(FlacDec::new());
    let probe = p.add(FormatProbe { seen: Arc::clone(&seen) });
    // filesrc(bytes) -> flacdec(bytes sink / audio/raw src, dynamic) -> probe(audio/raw).
    p.link((src, "src"), (dec, "sink")).expect("filesrc->flacdec");
    p.link((dec, "src"), (probe, "sink")).expect("flacdec->probe");
    p.run().expect("run");

    // The probe saw a concrete audio/raw format that existed only at runtime (STREAMINFO):
    // 44100 Hz, 2 channels, s16 — dynamic caps proven end to end.
    let f = seen
        .lock()
        .unwrap()
        .clone()
        .expect("probe received a FormatChange");
    assert_eq!(p.family_name(f.family), Some("audio/raw"), "announced family");
    let rate = p.field_id("rate").expect("rate interned");
    let channels = p.field_id("channels").expect("channels interned");
    let sample = p.field_id("sample").expect("sample interned");
    let s16 = p.value_id("s16").expect("s16 interned");
    assert_eq!(f.get(rate), Some(Value::Int(44100)), "runtime sample rate");
    assert_eq!(f.get(channels), Some(Value::Int(2)), "runtime channel count");
    assert_eq!(f.get(sample), Some(Value::Id(s16)), "runtime sample format");

    let _ = std::fs::remove_file(&path);
}
