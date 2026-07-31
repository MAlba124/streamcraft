//! `audiogain` — linear volume with a declick ramp (spec: Formats; Dynamic element properties;
//! flush/seek). What the in-module unit tests cannot prove, and this file does:
//!
//! 1. **The unity path really is a move.** [`unity_forwards_the_buffer_by_move`] compares the
//!    *pool-slot pointer* of the buffer that went in with the one that came out, so "copy-free"
//!    is checked rather than asserted — a copy would necessarily land in a different slot,
//!    because the input slot is still alive while `process()` allocates.
//! 2. **The arithmetic, through the real element shell.** Exact halving in `f32`, saturation at
//!    the integer full scale, and the partial-frame carry across ragged buffers.
//! 3. **The declick ramp.** That a live gain change and a mute are *interpolated*, with a
//!    bounded sample-to-sample slope, and that the ramp lands exactly on the target.
//! 4. **Flush.** That a seek drops both the carry and a half-finished ramp.

// Test setup, not an element hot path: signal generation, byte staging and the recording sink
// all allocate freely (the sanctioned whole-file exception in `clippy.toml`).
#![allow(clippy::disallowed_methods)]

use profluens_audio::format::{FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE};
use profluens_audio::{AudioGain, SampleFormat, GAIN_MAX, GAIN_MIN, RAMP_MS};
use profluens_core::event::Event;
use profluens_core::format::{Value, ValueDesc};
use profluens_core::harness::Harness;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

const RATE_HZ: u32 = 48_000;

// --- Helpers --------------------------------------------------------------------------

fn f32_bytes(input: &[f32]) -> Vec<u8> {
    input.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn take_f32(out: &mut Vec<f32>, bytes: &[u8]) {
    for w in bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*w));
    }
}

/// Ramp length in frames at [`RATE_HZ`] — the element's own `RAMP_MS` at the negotiated rate.
fn ramp_frames() -> usize {
    RATE_HZ as usize * RAMP_MS / 1000
}

fn rig(channels: u16, sample: SampleFormat, gain: f32) -> Harness {
    let mut h = Harness::with_slot_size(AudioGain::new(gain), 1 << 16);
    h.fix_format(
        "sink",
        "audio/raw",
        &[
            (FIELD_RATE, ValueDesc::Int(RATE_HZ as i64)),
            (FIELD_CHANNELS, ValueDesc::Int(channels as i64)),
            (FIELD_SAMPLE, ValueDesc::Id(sample.caps_name())),
        ],
    );
    h.start().expect("start");
    h
}

/// Push `bytes` in `chunk`-sized buffers and collect every output byte.
fn drive(h: &mut Harness, bytes: &[u8], chunk: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for part in bytes.chunks(chunk.max(1)) {
        let buf = h.alloc(part);
        h.push("sink", buf).expect("push");
        while let Some(b) = h.pull("src") {
            out.extend_from_slice(b.memory.data());
        }
    }
    for b in h.eos().expect("eos") {
        out.extend_from_slice(b.memory.data());
    }
    out
}

/// Push 100 ms of full-scale DC, apply `value` to `prop`, push another 100 ms, and return the
/// **steady-state** output level once the ramp has finished. Because the input is a constant
/// 1.0, that level *is* the multiplier the element settled on.
fn steady_level_after(prop: &'static str, value: Value) -> f32 {
    let mut h = rig(1, SampleFormat::F32, 1.0);
    let dc = vec![1.0f32; RATE_HZ as usize / 10];
    let buf = h.alloc(&f32_bytes(&dc));
    h.push("sink", buf).expect("push");
    while h.pull("src").is_some() {}

    h.push_event(Event::PropChanged { name: prop, value }).expect("prop event");

    let mut out = Vec::new();
    let buf = h.alloc(&f32_bytes(&dc));
    h.push("sink", buf).expect("push");
    while let Some(b) = h.pull("src") {
        take_f32(&mut out, b.memory.data());
    }
    *out.last().expect("output after the property change")
}

// --- 1. The unity fast path -----------------------------------------------------------

#[test]
fn unity_forwards_the_buffer_by_move() {
    // The pool slot that goes in is the pool slot that comes out. A copy could not do this:
    // the input buffer is still alive while `process()` runs, so `ctx.alloc` would have to
    // hand back a *different* slot.
    let mut h = rig(2, SampleFormat::F32, 1.0);
    let input: Vec<f32> = (0..2048).map(|i| (i as f32 / 512.0) - 2.0).collect();
    let bytes = f32_bytes(&input);

    let buf = h.alloc(&bytes);
    let ptr = buf.memory.data().as_ptr();
    h.push("sink", buf).expect("push");

    let out = h.pull("src").expect("one output buffer");
    assert_eq!(
        out.memory.data().as_ptr(),
        ptr,
        "unity must forward the pool slot itself — this output was copied into a new slot"
    );
    assert_eq!(out.memory.data(), &bytes[..], "and the payload is untouched");
    assert!(h.pull("src").is_none(), "one buffer in, one buffer out — no re-chunking");
}

#[test]
fn a_non_unity_gain_takes_the_copy_path() {
    // The contrast case, so the pointer check above cannot pass vacuously.
    let mut h = rig(2, SampleFormat::F32, 0.5);
    let bytes = f32_bytes(&vec![1.0f32; 2048]);
    let buf = h.alloc(&bytes);
    let ptr = buf.memory.data().as_ptr();
    h.push("sink", buf).expect("push");
    let out = h.pull("src").expect("one output buffer");
    assert_ne!(
        out.memory.data().as_ptr(),
        ptr,
        "a scaled buffer cannot be the input slot — it would have been scaled in place"
    );
}

#[test]
fn unity_passthrough_is_bit_exact_including_hostile_payloads() {
    // Unity skips the multiply entirely rather than multiplying by 1.0, so even a signalling
    // NaN survives with its payload intact.
    let mut h = rig(1, SampleFormat::F32, 1.0);
    let input = [0.0f32, -0.0, 1.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e-40];
    let bytes = f32_bytes(&input);
    let got = drive(&mut h, &bytes, 1 << 15);
    assert_eq!(got, bytes, "unity altered the payload");
}

// --- 2. Arithmetic --------------------------------------------------------------------

#[test]
fn gain_one_half_halves_every_f32_sample_exactly() {
    // The element is *constructed* at 0.5, so `start()` snaps rather than ramps — halving is
    // exact from the very first sample. And halving a normal f32 is exact in IEEE-754, so this
    // is a bit-for-bit comparison, not an epsilon one.
    let mut h = rig(2, SampleFormat::F32, 0.5);
    let input: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.37).sin() * 0.9).collect();
    let got = drive(&mut h, &f32_bytes(&input), 1 << 15);

    let mut out = Vec::new();
    take_f32(&mut out, &got);
    assert_eq!(out.len(), input.len(), "gain must not change the sample count");
    for (i, (a, b)) in input.iter().zip(&out).enumerate() {
        assert_eq!((a * 0.5).to_bits(), b.to_bits(), "sample {i}: {a} * 0.5 != {b}");
    }
}

#[test]
fn s16_saturates_at_gain_four_instead_of_wrapping() {
    let mut h = rig(1, SampleFormat::S16, 4.0);
    let input: [i16; 7] = [0, 1, -1, 8_192, -8_192, i16::MAX, i16::MIN];
    let bytes: Vec<u8> = input.iter().flat_map(|s| s.to_le_bytes()).collect();
    let got = drive(&mut h, &bytes, 1 << 15);

    // 8192*4 overshoots by one and pins; -8192*4 lands exactly on the negative rail; the rails
    // themselves stay put. A wrapping multiply would invert the sign of every one of these.
    let want: Vec<u8> = [0i16, 4, -4, 32_767, -32_768, 32_767, -32_768]
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    assert_eq!(got, want, "s16 at 4x must saturate at full scale, never wrap");
}

#[test]
fn every_sample_format_survives_a_round_trip() {
    // The whole `audio/raw` matrix goes through the same code path; silence must stay silence
    // and the byte count must be preserved in each.
    for (fmt, silence) in [
        (SampleFormat::U8, vec![128u8; 64]),
        (SampleFormat::S16, vec![0u8; 64]),
        (SampleFormat::S24, vec![0u8; 66]),
        (SampleFormat::S32, vec![0u8; 64]),
        (SampleFormat::F32, vec![0u8; 64]),
    ] {
        let mut h = rig(2, fmt, 0.5);
        let got = drive(&mut h, &silence, 1 << 15);
        assert_eq!(got.len(), silence.len(), "{fmt:?}: byte count changed");
        assert_eq!(got, silence, "{fmt:?}: scaled silence is not silence");
    }
}

#[test]
fn ragged_buffers_keep_the_interleave() {
    // 997 bytes is coprime with the 8-byte stereo f32 stride, so almost every buffer boundary
    // lands mid-frame and the one-frame carry has to reassemble it. Channels carry opposite
    // signs, so a desync would show up immediately as a sign flip.
    let mut h = rig(2, SampleFormat::F32, 0.5);
    let frames = 20_000usize;
    let input: Vec<f32> = (0..frames)
        .flat_map(|i| [0.5 + i as f32 * 1e-5, -(0.5 + i as f32 * 1e-5)])
        .collect();
    let got = drive(&mut h, &f32_bytes(&input), 997);

    let mut out = Vec::new();
    take_f32(&mut out, &got);
    assert_eq!(out.len(), input.len(), "ragged buffering changed the sample count");
    for (i, (a, b)) in input.iter().zip(&out).enumerate() {
        assert_eq!((a * 0.5).to_bits(), b.to_bits(), "sample {i}: the carry desynced the lanes");
    }
}

// --- 3. The declick ramp --------------------------------------------------------------

#[test]
fn mute_ramps_to_silence_then_emits_silence() {
    let mut h = rig(1, SampleFormat::F32, 1.0);
    // Full-scale DC, so the output sample value *is* the multiplier in force.
    let dc = vec![1.0f32; RATE_HZ as usize / 10];
    let buf = h.alloc(&f32_bytes(&dc));
    h.push("sink", buf).expect("push");
    while h.pull("src").is_some() {}

    h.push_event(Event::PropChanged { name: "mute", value: Value::Int(1) }).expect("mute");

    let mut out = Vec::new();
    let buf = h.alloc(&f32_bytes(&dc));
    h.push("sink", buf).expect("push");
    while let Some(b) = h.pull("src") {
        take_f32(&mut out, b.memory.data());
    }

    let ramp = ramp_frames();
    assert!(out.len() > ramp, "need output past the end of the ramp");
    assert_eq!(out[0], 1.0, "the ramp starts at the gain that was in force");
    // Monotone descent, no overshoot, no jump.
    for i in 1..ramp {
        assert!(out[i] <= out[i - 1], "frame {i}: the ramp went back up ({} -> {})", out[i - 1], out[i]);
        assert!((0.0..=1.0).contains(&out[i]), "frame {i}: ramp left [0, 1] at {}", out[i]);
    }
    assert!(out[ramp - 1] < 0.01, "the ramp did not reach silence in {ramp} frames");
    // …and then it is *exactly* zero, not merely small: the last step snaps onto the target.
    for (i, s) in out[ramp..].iter().enumerate() {
        assert_eq!(*s, 0.0, "frame {} after the ramp is not silent: {s}", ramp + i);
    }
}

#[test]
fn a_live_gain_change_has_no_discontinuity() {
    // Constant DC, so the *only* sample-to-sample change in the whole stream is the ramp
    // itself. Stepping the gain would put the entire 0.75 change into one sample boundary;
    // ramping spreads it over `ramp_frames()` and bounds the slope at 0.75/240 ≈ 0.003.
    let mut h = rig(1, SampleFormat::F32, 1.0);
    let dc = vec![1.0f32; RATE_HZ as usize / 10];

    let mut out = Vec::new();
    let buf = h.alloc(&f32_bytes(&dc));
    h.push("sink", buf).expect("push");
    while let Some(b) = h.pull("src") {
        take_f32(&mut out, b.memory.data());
    }
    let before = out.len();

    h.push_event(Event::PropChanged { name: "gain", value: Value::Rat(1, 4) }).expect("gain");

    let buf = h.alloc(&f32_bytes(&dc));
    h.push("sink", buf).expect("push");
    while let Some(b) = h.pull("src") {
        take_f32(&mut out, b.memory.data());
    }
    assert!(before > 0 && out.len() > before, "output on both sides of the change");

    let ideal = 0.75 / ramp_frames() as f32;
    let (worst, at) = out
        .windows(2)
        .enumerate()
        .map(|(i, w)| ((w[1] - w[0]).abs(), i))
        .fold((0.0f32, 0), |a, b| if b.0 > a.0 { b } else { a });
    assert!(
        worst <= ideal * 1.5,
        "discontinuity of {worst} at sample {at}: the gain stepped instead of ramping \
         (a single-sample step would be 0.75; the ramp's own slope is {ideal})"
    );
    assert_eq!(*out.last().unwrap(), 0.25, "the ramp settles exactly on the target");
}

#[test]
fn gain_prop_accepts_the_rational_and_integer_spellings() {
    // Two spellings of the same number must land on the same multiplier.
    assert_eq!(steady_level_after("gain", Value::Rat(1, 2)), 0.5);
    assert_eq!(steady_level_after("gain", Value::Rat(5, 10)), 0.5);
    assert_eq!(steady_level_after("gain", Value::Int(2)), 2.0);
    // Out-of-range requests clamp rather than failing.
    assert_eq!(steady_level_after("gain", Value::Int(99)), GAIN_MAX);
    assert_eq!(steady_level_after("gain", Value::Rat(-1, 1)), GAIN_MIN);
    // A zero denominator is rejected, leaving the gain where it was.
    assert_eq!(steady_level_after("gain", Value::Rat(1, 0)), 1.0);
}

#[test]
fn mute_prop_accepts_the_integer_spelling() {
    assert_eq!(steady_level_after("mute", Value::Int(1)), 0.0, "mute=1 silences");
    assert_eq!(
        steady_level_after("mute", Value::Int(0)),
        1.0,
        "mute=0 on an unmuted element is a no-op"
    );
}

// --- 4. Flush / seek ------------------------------------------------------------------

#[test]
fn flush_resets_the_ramp_and_the_carry() {
    // Two pieces of pre-seek state at once: a partial interchannel frame parked in the carry,
    // and a ramp half-way to a new gain. Post-seek audio must be at the target gain from its
    // very first sample, and must not be prefixed by the stale partial frame.
    let mut h = rig(2, SampleFormat::F32, 1.0);

    // 6 bytes is less than the 8-byte stereo f32 stride, so it lands wholly in the carry.
    let buf = h.alloc(&[0xAAu8; 6]);
    h.push("sink", buf).expect("push");
    assert!(h.pull("src").is_none(), "a partial frame produces no output");

    // Start a ramp, then seek before it can finish.
    h.push_event(Event::PropChanged { name: "gain", value: Value::Rat(1, 4) }).expect("gain");
    h.push_event(Event::FlushStart).expect("flush");

    // Post-seek: L = +1, R = -1, so a part-frame shift shows up as a sign flip.
    let frames = 4_096usize;
    let input: Vec<f32> = (0..frames).flat_map(|_| [1.0f32, -1.0]).collect();
    let got = drive(&mut h, &f32_bytes(&input), 1 << 15);

    let mut out = Vec::new();
    take_f32(&mut out, &got);
    assert_eq!(
        out.len(),
        input.len(),
        "post-flush output is longer than the input — the stale carry was prepended"
    );
    for (i, (a, b)) in input.iter().zip(&out).enumerate() {
        assert_eq!(
            *b,
            a * 0.25,
            "sample {i}: expected the target gain from the first sample, got {b} \
             (a surviving ramp or a shifted interleave)"
        );
    }
}

#[test]
fn flush_alone_does_not_change_the_configured_gain() {
    // A seek moves the read head; it must not touch the volume the user set.
    let mut h = rig(1, SampleFormat::F32, 0.5);
    h.push_event(Event::FlushStart).expect("flush");
    let got = drive(&mut h, &f32_bytes(&[1.0f32; 64]), 1 << 15);
    let mut out = Vec::new();
    take_f32(&mut out, &got);
    assert!(out.iter().all(|s| *s == 0.5), "the flush changed the gain");
}

// --- 5. Hostile input -----------------------------------------------------------------

#[test]
fn non_finite_samples_do_not_wedge_the_element() {
    // NaN/inf pass through the multiplier (float is the headroom format — see the module
    // docs). What must not happen is a panic, an error, or the ramp losing its place.
    let mut h = rig(2, SampleFormat::F32, 0.5);
    let poisoned: Vec<f32> = (0..8_000)
        .map(|i| match i % 4 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            _ => 0.5,
        })
        .collect();
    let clean = vec![1.0f32; 8_000];

    let mut out = Vec::new();
    for part in [&poisoned, &clean] {
        for chunk in f32_bytes(part).chunks(997) {
            let buf = h.alloc(chunk);
            h.push("sink", buf).expect("hostile input must not error");
            while let Some(b) = h.pull("src") {
                take_f32(&mut out, b.memory.data());
            }
        }
    }
    for b in h.eos().expect("eos") {
        take_f32(&mut out, b.memory.data());
    }

    assert_eq!(out.len(), poisoned.len() + clean.len(), "frames went missing");
    // The clean tail is still scaled correctly: no NaN leaked into the gain state.
    for (i, s) in out[poisoned.len()..].iter().enumerate() {
        assert_eq!(*s, 0.5, "clean sample {i} after a NaN burst came out as {s}");
    }
}

#[test]
fn absurd_buffer_shapes_are_survivable() {
    let mut h = rig(2, SampleFormat::F32, 0.5);
    // Empty and sub-frame buffers: never a whole frame, so they only feed the carry.
    for len in [0usize, 1, 3, 7] {
        let buf = h.alloc(&vec![0u8; len]);
        h.push("sink", buf).expect("push");
    }
    // A real run still works afterwards.
    let got = drive(&mut h, &f32_bytes(&vec![1.0f32; 4_096]), 1 << 15);
    assert!(!got.is_empty(), "the element wedged on ragged input");
}

// --- 6. Timestamps --------------------------------------------------------------------

#[test]
fn pts_is_carried_across_the_gain_stage() {
    // A volume control must not destroy timing: a clock-driven sink downstream schedules on
    // these stamps.
    let mut h = rig(1, SampleFormat::F32, 0.5);
    let mut buf = h.alloc(&f32_bytes(&vec![1.0f32; 4_096]));
    buf.pts = Timestamp::from_secs(10);
    h.push("sink", buf).expect("push");
    let out = h.pull("src").expect("output");
    assert_eq!(out.pts, Timestamp::from_secs(10), "pts was dropped or rewritten");
    assert_eq!(
        out.duration,
        Timestamp::from_nanos(4_096 * 1_000_000_000 / RATE_HZ as u64),
        "duration must span the frames in the buffer"
    );
}

// --- 7. A real pipeline, and the decimal-text spelling ---------------------------------

/// A source with concrete `audio/raw` caps (f32 mono @ 48 kHz), so link-time negotiation fixes
/// them on the gain stage's sink and it can infer its format.
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

    static FIELDS: [FieldDesc; 3] = [
        FieldDesc {
            field: FIELD_RATE,
            allowed: ConstraintDesc::Eq(ValueDesc::Int(48_000)),
            preferred: None,
        },
        FieldDesc {
            field: FIELD_CHANNELS,
            allowed: ConstraintDesc::Eq(ValueDesc::Int(1)),
            preferred: None,
        },
        FieldDesc {
            field: FIELD_SAMPLE,
            allowed: ConstraintDesc::Eq(ValueDesc::Id("f32")),
            preferred: None,
        },
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
            let Some(mut buf) = ctx.try_alloc(PadId(0)) else { return Ok(Flow::Ok) };
            let cap = buf.memory.capacity() / 4 * 4; // whole mono f32 frames
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

/// Accumulates every payload byte the sink receives.
mod record_sink {
    use std::sync::{Arc, Mutex};

    use profluens_core::batch::Inputs;
    use profluens_core::ctx::Ctx;
    use profluens_core::element::{
        Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
    };
    use profluens_core::error::Error;
    use profluens_core::event::Event;
    use profluens_core::format::OfferDesc;
    use profluens_core::time::Timestamp;

    pub type Log = Arc<Mutex<Vec<u8>>>;

    static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
    static PADS: [PadDesc; 1] = [PadDesc {
        name: "sink",
        direction: Direction::Sink,
        offers: &OFFERS,
        dynamic: false,
        validate: None,
    }];
    static DESC: ElementDesc = ElementDesc {
        name: "recordsink",
        pads: &PADS,
        props: &[],
        sched: SchedHint::Passive,
        inputs: InputPolicy::Single,
        latency: LatencyDesc {
            min: Timestamp::ZERO,
            max: Timestamp::ZERO,
            is_live: false,
            jitter: Timestamp::ZERO,
        },
        make_default: None,
    };

    pub struct RecordSink(Log);
    impl RecordSink {
        pub fn new() -> (Self, Log) {
            let log: Log = Arc::new(Mutex::new(Vec::new()));
            (Self(Arc::clone(&log)), log)
        }
    }
    impl Element for RecordSink {
        fn desc(&self) -> &'static ElementDesc {
            &DESC
        }
        fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
            Ok(())
        }
        fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
            while let Some(b) = inputs.pop() {
                self.0.lock().unwrap().extend_from_slice(b.memory.data());
            }
            Ok(Flow::Ok)
        }
        fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
            Ok(())
        }
        fn stop(&mut self, _ctx: &mut Ctx) {}
    }
}

/// `rawaudiosrc(f32 mono 48k) ! audiogain ! recordsink`, with `props` applied as **strings** —
/// the launch-string path, which interns them as `Value::Id`.
fn run_pipeline(input: &[f32], props: &[(&str, &str)]) -> Vec<f32> {
    let mut p = Pipeline::new();
    p.set_pool(1 << 15, 32);
    let src = p.add(raw_src::RawAudioSrc::new(f32_bytes(input)));
    let gain = p.add(AudioGain::new(1.0));
    let (sink, log) = record_sink::RecordSink::new();
    let snk = p.add(sink);
    p.link((src, "src"), (gain, "sink")).expect("link src->audiogain");
    p.link((gain, "src"), (snk, "sink")).expect("link audiogain->sink");
    for (name, value) in props {
        p.set_str(gain, name, value).unwrap_or_else(|e| panic!("{name}={value}: {e:?}"));
    }
    p.run().expect("run");

    let bytes = log.lock().unwrap().clone();
    let mut out = Vec::new();
    take_f32(&mut out, &bytes);
    out
}

#[test]
fn pipeline_applies_a_decimal_text_gain() {
    // The third spelling: `set_str` interns "0.5" exactly as the launch-string parser does, and
    // the element resolves it back through `ctx.value_name`. Applied in `start()`, so it snaps
    // — every sample is halved, including the first.
    let input = vec![1.0f32; 48_000];
    let out = run_pipeline(&input, &[("gain", "0.5")]);
    assert_eq!(out.len(), input.len(), "the pipeline lost samples");
    assert!(out.iter().all(|s| *s == 0.5), "gain=\"0.5\" did not resolve to a half");
}

#[test]
fn pipeline_applies_a_text_mute() {
    // `mute=true` is what a launch string produces; the workspace's `Value::Int` decoders would
    // silently ignore it, so this element accepts the text form too.
    let input = vec![1.0f32; 48_000];
    let out = run_pipeline(&input, &[("mute", "true")]);
    assert_eq!(out.len(), input.len());
    assert!(out.iter().all(|s| *s == 0.0), "mute=\"true\" did not silence the stream");
    // …and the false spelling leaves it alone.
    let out = run_pipeline(&input, &[("mute", "false")]);
    assert!(out.iter().all(|s| *s == 1.0), "mute=\"false\" silenced the stream");
}

#[test]
fn registry_knows_audiogain() {
    let mut reg = profluens_core::registry::Registry::new();
    profluens_audio::register(&mut reg);
    let desc = reg.get("audiogain").expect("audiogain is registered");
    assert_eq!(desc.name, "audiogain");
    assert!(desc.make_default.is_some(), "constructible from a launch string");
    for name in ["gain", "mute"] {
        let prop = desc.props.iter().find(|p| p.name == name).expect("declared prop");
        assert!(prop.live, "{name} must be settable while playing");
    }

    let mut p = Pipeline::new();
    reg.parse(&mut p, "audiogain gain=0.85 mute=0").expect("parse audiogain gain=0.85 mute=0");
}
