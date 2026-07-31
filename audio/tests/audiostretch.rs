//! `audiostretch` — the PICOLA time stretcher (spec: Formats; Dynamic element properties;
//! flush/seek). Three things are proven here that the in-module unit tests cannot:
//!
//! 1. **Fidelity to the original.** The DSP is a port of `mskp-stretch` (the author's music
//!    player). [`golden_vectors_match_the_original_bit_exactly`] pins the port to the
//!    original's *exact* output — sample count and every sample bit — at four rates, two
//!    sample rates, mono and stereo.
//! 2. **The push-model shell.** Ragged buffers, live rate changes at batch boundaries, flush,
//!    and the EOS tail drain, all through the real element harness.
//! 3. **The timestamp policy.** Outgoing `pts` is a continuous playback-time grid:
//!    `pts[i] + duration[i] == pts[i+1]` exactly, across buffer boundaries *and* across a
//!    mid-stream rate change.
//!
//! # Regenerating the golden constants
//!
//! The goldens are `(output_len, FNV-1a64 over the output f32 bit patterns)` produced by the
//! **original** crate. They are constants rather than embedded waveforms because a single
//! vector is ~1.5 MB. To regenerate (e.g. after intentionally changing the DSP), build the
//! two-sided comparison harness — a crate depending on both `mskp-stretch` (path:
//! `musikkspiller/stretch`) and `profluens-audio` — that generates the input with the
//! [`gen_input`] function copied verbatim from this file, runs the original's
//! `TimeStretch::new(src, rate).collect()` and this element side by side, and prints
//! `len`/digest plus max-abs-diff per rate. That harness reports **max abs diff = 0.0,
//! BIT-EXACT** for every row in [`GOLDEN`].
//!
//! [`gen_input`] deliberately uses no transcendental functions: every operation is an IEEE-754
//! add/sub/mul/div/abs, and the stretcher itself is likewise transcendental-free, so these
//! digests are reproducible on any platform rather than pinned to one libm.

// Test setup, not an element hot path: signal generation, byte staging, and the recording
// sink all allocate freely (the sanctioned whole-file exception in `clippy.toml`).
#![allow(clippy::disallowed_methods)]

use profluens_audio::format::{FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE};
use profluens_audio::{AudioStretch, SampleFormat};
use profluens_core::element::Flow;
use profluens_core::event::Event;
use profluens_core::format::{Value, ValueDesc};
use profluens_core::harness::Harness;
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;

// --- Deterministic test signal (see the module docs) ----------------------------------

/// Speech-like: a ~120 Hz glottal pulse train (a squared triangle — a real pitch period for
/// the AMDF search to lock onto), a slow triangular envelope that walks through loud/quiet so
/// the voiced/unvoiced hysteresis is exercised, and per-sample LCG noise so no two frames are
/// identical. Channels are decorrelated by a DC offset.
fn gen_input(rate_hz: u32, channels: usize, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames * channels);
    let mut seed: u32 = 0x1234_5678;
    let period = (rate_hz / 120) as usize;
    let eperiod = (rate_hz * 7 / 10) as usize;
    for n in 0..frames {
        let phase = (n % period) as f32 / period as f32;
        let tri = 1.0 - (2.0 * phase - 1.0).abs();
        let voiced = tri * tri * 2.0 - 0.5;
        let ephase = (n % eperiod) as f32 / eperiod as f32;
        let env = 1.0 - (2.0 * ephase - 1.0).abs();
        for c in 0..channels {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (seed >> 9) as f32 / 4_194_304.0 - 1.0;
            out.push(env * voiced * 0.6 + 0.05 * noise - 0.02 * c as f32);
        }
    }
    out
}

fn fnv1a64(samples: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for s in samples {
        for b in s.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn f32_bytes(input: &[f32]) -> Vec<u8> {
    input.iter().flat_map(|s| s.to_ne_bytes()).collect()
}

fn take_f32(out: &mut Vec<f32>, bytes: &[u8]) {
    for w in bytes.as_chunks::<4>().0 {
        out.push(f32::from_ne_bytes(*w));
    }
}

// --- Harness driver -------------------------------------------------------------------

fn rig(rate_hz: u32, channels: u16, rate: f32) -> Harness {
    let mut h = Harness::with_slot_size(AudioStretch::new(rate), 1 << 16);
    h.fix_format(
        "sink",
        "audio/raw",
        &[
            (FIELD_RATE, ValueDesc::Int(rate_hz as i64)),
            (FIELD_CHANNELS, ValueDesc::Int(channels as i64)),
            (FIELD_SAMPLE, ValueDesc::Id(SampleFormat::F32.caps_name())),
        ],
    );
    h.start().expect("start");
    h
}

/// Push `input` through the element in `chunk_bytes`-sized buffers (a value that is *not* a
/// multiple of the frame stride exercises the partial-frame carry), then EOS. Returns the
/// output samples and the per-buffer `(pts, duration)` in emission order.
fn run(
    rate_hz: u32,
    channels: u16,
    rate: f32,
    input: &[f32],
    chunk_bytes: usize,
) -> (Vec<f32>, Vec<(Timestamp, Timestamp)>) {
    let mut h = rig(rate_hz, channels, rate);
    let bytes = f32_bytes(input);
    let mut out = Vec::new();
    let mut stamps = Vec::new();
    for chunk in bytes.chunks(chunk_bytes.max(1)) {
        let buf = h.alloc(chunk);
        h.push("sink", buf).expect("push");
        while let Some(b) = h.pull("src") {
            stamps.push((b.pts, b.duration));
            take_f32(&mut out, b.memory.data());
        }
    }
    for b in h.eos().expect("eos") {
        stamps.push((b.pts, b.duration));
        take_f32(&mut out, b.memory.data());
    }
    (out, stamps)
}

// --- 1. The fidelity gate -------------------------------------------------------------

/// `(sample_rate, channels, input_digest, [(rate, golden_len, golden_digest); 4])`.
/// Every row was produced by the ORIGINAL `mskp-stretch::TimeStretch`.
#[allow(clippy::type_complexity)]
const GOLDEN: [(u32, u16, u64, [(f32, usize, u64); 4]); 3] = [
    (
        44_100,
        2,
        0x0b47_5d41_0cfb_b3e8,
        [
            (0.75, 351_754, 0xb12c_d620_7fa2_1f53),
            (1.25, 211_978, 0xfd88_7187_523d_7277),
            (1.50, 177_446, 0xcddf_4e2b_e517_7950),
            (2.00, 133_902, 0xea0b_895a_9df7_d0f6),
        ],
    ),
    (
        48_000,
        2,
        0x6770_b4dd_90ae_010d,
        [
            (0.75, 383_214, 0x300b_0a3a_f89d_f1d7),
            (1.25, 230_518, 0x1474_deed_d0b7_5429),
            (1.50, 192_786, 0x09d8_c6f6_8d55_3cb5),
            (2.00, 145_630, 0xdceb_87e6_13d9_0088),
        ],
    ),
    (
        48_000,
        1,
        0x31ef_8128_0305_995d,
        [
            (0.75, 191_362, 0x5841_d4d5_8218_8918),
            (1.25, 115_446, 0x3330_ebf3_d483_c16b),
            (1.50, 96_638, 0x4c42_d1db_d78b_4e6c),
            (2.00, 73_011, 0xc6f7_f939_329c_4819),
        ],
    ),
];

#[test]
fn golden_vectors_match_the_original_bit_exactly() {
    for &(rate_hz, channels, in_digest, rows) in &GOLDEN {
        let frames = rate_hz as usize * 3; // 3 seconds
        let input = gen_input(rate_hz, channels as usize, frames);
        assert_eq!(
            fnv1a64(&input),
            in_digest,
            "{rate_hz} Hz {channels} ch: the test signal itself changed — the goldens below \
             describe a different input and must be regenerated (see the module docs)"
        );
        for &(rate, golden_len, golden_digest) in &rows {
            // Frame-aligned buffers, the ordinary case.
            let (out, _) = run(rate_hz, channels, rate, &input, 4096 * channels as usize * 4);
            assert_eq!(
                out.len(),
                golden_len,
                "{rate_hz} Hz {channels} ch rate={rate}: output length differs from the \
                 original's ({} vs {golden_len})",
                out.len()
            );
            assert_eq!(
                fnv1a64(&out),
                golden_digest,
                "{rate_hz} Hz {channels} ch rate={rate}: same length but different samples — \
                 the port is no longer bit-exact against mskp-stretch"
            );
        }
    }
}

#[test]
fn ragged_buffers_are_transparent() {
    // 997 bytes is coprime with every frame stride here, so almost every buffer boundary lands
    // mid-frame and the one-frame carry has to reassemble it. Output must not budge.
    let (rate_hz, channels) = (48_000u32, 2u16);
    let input = gen_input(rate_hz, channels as usize, rate_hz as usize * 3);
    for &(rate, golden_len, golden_digest) in &GOLDEN[1].3 {
        let (out, _) = run(rate_hz, channels, rate, &input, 997);
        assert_eq!(out.len(), golden_len, "rate={rate}: ragged buffering changed the length");
        assert_eq!(
            fnv1a64(&out),
            golden_digest,
            "rate={rate}: ragged buffering changed the samples — the partial-frame carry \
             desynced the channel lanes"
        );
    }
}

// --- 2. Passthrough -------------------------------------------------------------------

#[test]
fn rate_one_is_bit_exact_passthrough() {
    for &(rate_hz, channels) in &[(44_100u32, 2u16), (48_000, 1)] {
        let input = gen_input(rate_hz, channels as usize, rate_hz as usize);
        let (out, _) = run(rate_hz, channels, 1.0, &input, 4096 * channels as usize * 4);
        assert_eq!(out.len(), input.len(), "bypass must not change the sample count");
        for (i, (a, b)) in input.iter().zip(&out).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "bypass altered sample {i}");
        }
    }
}

#[test]
fn near_one_rates_also_bypass() {
    // Within BYPASS_EPSILON (0.02) the stretcher is skipped entirely — the original's rule.
    let input = gen_input(48_000, 2, 48_000);
    let (out, _) = run(48_000, 2, 1.01, &input, 1 << 15);
    assert_eq!(out.len(), input.len());
    assert!(input.iter().zip(&out).all(|(a, b)| a.to_bits() == b.to_bits()));
}

// --- 3. Timestamp policy --------------------------------------------------------------

#[test]
fn pts_is_a_continuous_playback_time_grid() {
    let input = gen_input(48_000, 2, 48_000 * 2);
    let (out, stamps) = run(48_000, 2, 1.5, &input, 1 << 15);
    assert!(stamps.len() > 2, "expected several output buffers");

    for w in stamps.windows(2) {
        let (pts, dur) = w[0];
        let (next, _) = w[1];
        assert!(pts.is_some() && dur.is_some(), "audiostretch must stamp pts and duration");
        assert_eq!(
            pts.saturating_add(dur),
            next,
            "gap or overlap in the playback-time grid: {pts:?} + {dur:?} != {next:?}"
        );
    }
    // The grid spans exactly the emitted frames at the negotiated rate.
    let (last_pts, last_dur) = *stamps.last().unwrap();
    let frames = (out.len() / 2) as u64;
    assert_eq!(
        last_pts.saturating_add(last_dur),
        Timestamp::from_nanos(frames * 1_000_000_000 / 48_000),
        "end of the grid must equal emitted_frames / sample_rate"
    );
    // …and 1.5x of 2 s of input is ~1.333 s of playback.
    let secs = last_pts.saturating_add(last_dur).nanos().unwrap() as f64 / 1e9;
    assert!((secs - 2.0 / 1.5).abs() < 0.02, "playback duration {secs}s, expected ~1.333s");
}

#[test]
fn pts_base_is_adopted_from_the_first_input_buffer() {
    let input = gen_input(48_000, 1, 48_000);
    let mut h = rig(48_000, 1, 1.5);
    let bytes = f32_bytes(&input);
    let mut first = None;
    for chunk in bytes.chunks(1 << 15) {
        let mut buf = h.alloc(chunk);
        buf.pts = Timestamp::from_secs(10); // upstream is 10 s into the stream
        h.push("sink", buf).expect("push");
        while let Some(b) = h.pull("src") {
            first.get_or_insert(b.pts);
        }
    }
    assert_eq!(
        first.expect("some output"),
        Timestamp::from_secs(10),
        "the first output pts must be the origin taken from upstream, not zero"
    );
}

#[test]
fn source_position_tracks_input_time_not_playback_time() {
    // The port of mskp-stretch's `Control::position()`, which is what the music player's
    // `playback_pos()` reads: at 2x, one second of *playback* is two seconds of episode.
    let el = AudioStretch::new(2.0);
    let pos = el.position();
    assert_eq!(pos.rate(), 2.0);

    let mut h = Harness::with_slot_size(el, 1 << 16);
    h.fix_format(
        "sink",
        "audio/raw",
        &[
            (FIELD_RATE, ValueDesc::Int(48_000)),
            (FIELD_CHANNELS, ValueDesc::Int(1)),
            (FIELD_SAMPLE, ValueDesc::Id("f32")),
        ],
    );
    h.start().expect("start");

    let input = gen_input(48_000, 1, 48_000 * 4); // 4 s of source
    let bytes = f32_bytes(&input);
    let mut out_frames = 0u64;
    for chunk in bytes.chunks(1 << 15) {
        let buf = h.alloc(chunk);
        h.push("sink", buf).expect("push");
        while let Some(b) = h.pull("src") {
            out_frames += (b.memory.data().len() / 4) as u64;
        }
    }
    for b in h.eos().expect("eos") {
        out_frames += (b.memory.data().len() / 4) as u64;
    }

    let source_secs = pos.position().nanos().unwrap() as f64 / 1e9;
    let playback_secs = out_frames as f64 / 48_000.0;
    assert!(
        (source_secs - 4.0).abs() < 0.05,
        "source position {source_secs}s, expected ~4.0s (all of the input)"
    );
    assert!(
        (playback_secs - 2.0).abs() < 0.05,
        "playback duration {playback_secs}s, expected ~2.0s at 2x"
    );
    assert_eq!(pos.source_frames(), 48_000 * 4, "every input frame accounted for");
    assert_eq!(pos.output_frames(), out_frames);
}

// --- 4. Live rate changes -------------------------------------------------------------

/// A slow triangle: smooth, with a known maximum legitimate sample-to-sample slope, so a
/// splice discontinuity (a click) shows up as an outlier delta.
fn triangle(rate_hz: u32, frames: usize, hz: usize) -> Vec<f32> {
    let period = rate_hz as usize / hz;
    (0..frames)
        .map(|n| {
            let phase = (n % period) as f32 / period as f32;
            0.8 * (2.0 * (1.0 - (2.0 * phase - 1.0).abs()) - 1.0)
        })
        .collect()
}

#[test]
fn rate_change_mid_stream_keeps_pts_and_waveform_continuous() {
    let rate_hz = 48_000u32;
    let input = triangle(rate_hz, rate_hz as usize * 2, 100);
    let mut h = rig(rate_hz, 1, 1.0);
    let bytes = f32_bytes(&input);
    let half = bytes.len() / 2 / 4 * 4;

    let mut out = Vec::new();
    let mut stamps = Vec::new();
    let drain = |h: &mut Harness, out: &mut Vec<f32>, stamps: &mut Vec<_>| {
        while let Some(b) = h.pull("src") {
            stamps.push((b.pts, b.duration));
            take_f32(out, b.memory.data());
        }
    };

    for chunk in bytes[..half].chunks(1 << 15) {
        let buf = h.alloc(chunk);
        h.push("sink", buf).expect("push");
        drain(&mut h, &mut out, &mut stamps);
    }
    let before = out.len();

    // The live knob: an exact rational, delivered the way the scheduler delivers it.
    h.push_event(Event::PropChanged { name: "rate", value: Value::Rat(5, 2) })
        .expect("rate prop event");

    for chunk in bytes[half..].chunks(1 << 15) {
        let buf = h.alloc(chunk);
        h.push("sink", buf).expect("push");
        drain(&mut h, &mut out, &mut stamps);
    }
    for b in h.eos().expect("eos") {
        stamps.push((b.pts, b.duration));
        take_f32(&mut out, b.memory.data());
    }

    // 1 s at 1.0 + 1 s at 2.5 = 1.4 s of playback.
    let secs = out.len() as f64 / rate_hz as f64;
    assert!((secs - 1.4).abs() < 0.05, "playback duration {secs}s, expected ~1.4s");
    assert!(before > 0 && out.len() > before, "output on both sides of the change");

    // No pts discontinuity: the grid must not jump when the rate does.
    for w in stamps.windows(2) {
        assert_eq!(
            w[0].0.saturating_add(w[0].1),
            w[1].0,
            "the rate change tore the playback-time grid"
        );
    }

    // No click: a 100 Hz triangle at 48 kHz slopes 0.8*4/480 ≈ 0.0067 per sample, with one
    // legitimate corner per half-period. Ten times that would be an audible splice.
    let slope = 0.8 * 4.0 / (rate_hz as f32 / 100.0);
    let (worst, at) = out
        .windows(2)
        .enumerate()
        .map(|(i, w)| ((w[1] - w[0]).abs(), i))
        .fold((0.0f32, 0), |a, b| if b.0 > a.0 { b } else { a });
    assert!(
        worst < 10.0 * slope,
        "discontinuity of {worst} at sample {at} (natural slope {slope}) — the splice clicked"
    );
}

#[test]
fn rate_prop_accepts_all_three_spellings() {
    let input = gen_input(48_000, 1, 48_000);
    let golden = run(48_000, 1, 1.5, &input, 1 << 15).0;
    for value in [Value::Rat(3, 2), Value::Rat(15, 10)] {
        let mut h = rig(48_000, 1, 1.0);
        h.push_event(Event::PropChanged { name: "rate", value }).expect("rate event");
        let bytes = f32_bytes(&input);
        let mut out = Vec::new();
        for chunk in bytes.chunks(1 << 15) {
            let buf = h.alloc(chunk);
            h.push("sink", buf).expect("push");
            while let Some(b) = h.pull("src") {
                take_f32(&mut out, b.memory.data());
            }
        }
        for b in h.eos().expect("eos") {
            take_f32(&mut out, b.memory.data());
        }
        assert_eq!(out.len(), golden.len(), "{value:?} did not resolve to 1.5");
    }
    // Out-of-range requests clamp instead of failing.
    assert_eq!(AudioStretch::new(10.0).rate(), profluens_audio::RATE_MAX);
    assert_eq!(AudioStretch::new(0.1).rate(), profluens_audio::RATE_MIN);
}

// --- 5. Flush / seek ------------------------------------------------------------------

#[test]
fn flush_drops_every_trace_of_pre_seek_audio() {
    let rate_hz = 48_000u32;
    let mut h = rig(rate_hz, 1, 2.0);

    // Pre-seek: solid +1.0.
    let pre = vec![1.0f32; rate_hz as usize / 2];
    let buf = h.alloc(&f32_bytes(&pre));
    h.push("sink", buf).expect("push");
    while h.pull("src").is_some() {}

    h.push_event(Event::FlushStart).expect("flush");

    // Post-seek: solid -1.0. Any surviving pre-seek sample would show up as a crossfade
    // between +1 and -1, i.e. a value nowhere near -1.
    let post = vec![-1.0f32; rate_hz as usize / 2];
    let mut out = Vec::new();
    let buf = h.alloc(&f32_bytes(&post));
    h.push("sink", buf).expect("push");
    while let Some(b) = h.pull("src") {
        take_f32(&mut out, b.memory.data());
    }
    for b in h.eos().expect("eos") {
        take_f32(&mut out, b.memory.data());
    }
    assert!(!out.is_empty(), "expected output after the flush");
    for (i, s) in out.iter().enumerate() {
        assert!(*s < -0.9, "sample {i} = {s}: pre-flush audio blended into post-flush output");
    }
    // The playback-time grid restarts at the seek target (none set here, so zero).
    assert_eq!(
        run(rate_hz, 1, 2.0, &post, 1 << 15).1[0].0,
        Timestamp::ZERO,
        "a fresh stream starts its grid at the base"
    );
}

// --- 6. EOS tail ----------------------------------------------------------------------

/// Frames a splice round holds back — `3 * max_period`, `max_period = rate / 65`.
fn window(rate_hz: u32) -> usize {
    3 * (rate_hz as usize / 65)
}

#[test]
fn eos_drains_the_final_window() {
    let rate_hz = 48_000u32;
    let frames = 20_000usize;
    let el = AudioStretch::new(1.5);
    let pos = el.position();

    let mut h = Harness::with_slot_size(el, 1 << 16);
    h.fix_format(
        "sink",
        "audio/raw",
        &[
            (FIELD_RATE, ValueDesc::Int(rate_hz as i64)),
            (FIELD_CHANNELS, ValueDesc::Int(1)),
            (FIELD_SAMPLE, ValueDesc::Id("f32")),
        ],
    );
    h.start().expect("start");

    let input = gen_input(rate_hz, 1, frames);
    let mut before = 0usize;
    for chunk in f32_bytes(&input).chunks(1 << 15) {
        let buf = h.alloc(chunk);
        h.push("sink", buf).expect("push");
        while let Some(b) = h.pull("src") {
            before += b.memory.data().len() / 4;
        }
    }
    let held = frames as u64 - pos.source_frames();
    assert!(held > 0, "mid-stream the element holds a search window back");

    let mut after = before;
    for b in h.eos().expect("eos") {
        after += b.memory.data().len() / 4;
    }
    assert!(after > before, "EOS must flush the held-back search window ({before} -> {after})");
    assert_eq!(
        pos.source_frames(),
        frames as u64,
        "after the drain every input frame is accounted for"
    );

    // PICOLA passes that final window through *verbatim* rather than splicing it — the
    // original's behaviour, inherited deliberately ("tempo drifts for the final few tens of
    // ms; inaudible"). So the tail is un-stretched and the output lands between "everything
    // stretched" and "everything stretched but the tail".
    let w = window(rate_hz);
    let lo = (frames - w) as f64 / 1.5;
    assert!(
        (lo..=lo + w as f64 + 2.0).contains(&(after as f64)),
        "{after} frames out of {frames} at 1.5x: outside [{lo}, {}] (stretched body + \
         un-stretched tail of at most {w} frames)",
        lo + w as f64
    );
}

// --- 7. Hostile input -----------------------------------------------------------------

#[test]
fn non_finite_samples_do_not_wedge_the_state_machine() {
    let rate_hz = 48_000u32;
    let clean = gen_input(rate_hz, 2, 20_000);
    let mut poisoned = gen_input(rate_hz, 2, 20_000);
    for (i, s) in poisoned.iter_mut().enumerate() {
        *s = match i % 4 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            _ => *s,
        };
    }

    let mut h = rig(rate_hz, 2, 1.75);
    let mut frames = 0usize;
    for part in [&poisoned, &clean] {
        for chunk in f32_bytes(part).chunks(1 << 15) {
            let buf = h.alloc(chunk);
            h.push("sink", buf).expect("hostile input must not error");
            while let Some(b) = h.pull("src") {
                frames += b.memory.data().len() / 8;
            }
        }
    }
    for b in h.eos().expect("eos") {
        frames += b.memory.data().len() / 8;
    }
    // The machine kept its schedule: 40 000 input frames at 1.75x is ~22 857 out. A wedged
    // splice length would show up as a wildly wrong count (or a hang, or a panic).
    let ratio = frames as f64 / (40_000.0 / 1.75);
    assert!(
        (0.95..=1.05).contains(&ratio),
        "NaN/inf derailed the splice schedule: {frames} frames out of 40000 in at 1.75x"
    );
}

#[test]
fn absurd_buffer_shapes_are_survivable() {
    let mut h = rig(48_000, 2, 1.5);
    // Empty and single-byte buffers: never a whole frame, so they only feed the carry.
    for len in [0usize, 1, 3, 7] {
        let buf = h.alloc(&vec![0u8; len]);
        assert_eq!(h.push("sink", buf).expect("push"), Flow::Ok);
    }
    // Then a real run still works.
    let input = gen_input(48_000, 2, 10_000);
    for chunk in f32_bytes(&input).chunks(1 << 15) {
        let buf = h.alloc(chunk);
        h.push("sink", buf).expect("push");
        while h.pull("src").is_some() {}
    }
    h.eos().expect("eos");
}

// --- 8. A real pipeline ---------------------------------------------------------------

/// A source with concrete `audio/raw` caps (f32 stereo @ 48 kHz), so link-time negotiation
/// fixes them on the stretcher's sink and it can infer its format — mirrors the `rawaudiosrc`
/// in `audioresample.rs`.
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

    pub const RATE: i64 = 48_000;
    pub const CHANNELS: i64 = 2;

    static FIELDS: [FieldDesc; 3] = [
        FieldDesc {
            field: FIELD_RATE,
            allowed: ConstraintDesc::Eq(ValueDesc::Int(RATE)),
            preferred: None,
        },
        FieldDesc {
            field: FIELD_CHANNELS,
            allowed: ConstraintDesc::Eq(ValueDesc::Int(CHANNELS)),
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
            let cap = buf.memory.capacity() / 8 * 8; // whole stereo f32 frames
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

/// Records `(pts, duration, bytes)` per buffer so the test can check the timestamp policy.
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

    pub type Log = Arc<Mutex<Vec<(Timestamp, Timestamp, usize)>>>;

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
                self.0.lock().unwrap().push((b.pts, b.duration, b.memory.data().len()));
            }
            Ok(Flow::Ok)
        }
        fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
            Ok(())
        }
        fn stop(&mut self, _ctx: &mut Ctx) {}
    }
}

#[test]
fn pipeline_rawaudiosrc_to_audiostretch_to_sink() {
    // `rawaudiosrc(f32 stereo 48k) ! audiostretch rate=3/2 ! recordsink`. The rate arrives via
    // the property (the element is built at 1.0), proving the parse/`Pipeline::set` path.
    let frames = 96_000usize; // 2 s
    let input = gen_input(48_000, 2, frames);
    let pcm = f32_bytes(&input);

    let mut p = Pipeline::new();
    p.set_pool(1 << 15, 32);
    let src = p.add(raw_src::RawAudioSrc::new(pcm));
    let stretch = p.add(AudioStretch::new(1.0));
    let (sink, log) = record_sink::RecordSink::new();
    let snk = p.add(sink);
    p.link((src, "src"), (stretch, "sink")).expect("link src->stretch");
    p.link((stretch, "src"), (snk, "sink")).expect("link stretch->sink");
    p.set(stretch, "rate", Value::Rat(3, 2)).expect("rate=3/2 accepted");
    p.run().expect("run");

    let log = log.lock().unwrap();
    assert!(!log.is_empty(), "the sink received nothing");
    let out_frames: usize = log.iter().map(|(_, _, n)| n / 8).sum();
    // input / 1.5, allowing for the un-stretched final search window (see
    // `eos_drains_the_final_window`), which is ~0.8 % of a 2 s stream.
    let expected = frames as f64 / 1.5;
    let ratio = out_frames as f64 / expected;
    assert!(
        (0.99..=1.02).contains(&ratio),
        "output {out_frames} frames, expected ~{expected} (input / 1.5)"
    );

    // Timestamp policy end to end: monotonic, gapless, and spanning input/1.5 of playback.
    let mut prev: Option<(Timestamp, Timestamp)> = None;
    for &(pts, dur, _) in log.iter() {
        assert!(pts.is_some(), "every buffer carries a pts");
        if let Some((p0, d0)) = prev {
            assert!(pts >= p0, "pts went backwards");
            assert_eq!(p0.saturating_add(d0), pts, "gap in the playback-time grid");
        }
        prev = Some((pts, dur));
    }
    let (last_pts, last_dur) = prev.unwrap();
    let secs = last_pts.saturating_add(last_dur).nanos().unwrap() as f64 / 1e9;
    assert!(
        (secs - 2.0 / 1.5).abs() < 0.03,
        "playback span {secs}s, expected ~1.333s for 2 s of input at 1.5x"
    );
}

#[test]
fn registry_knows_audiostretch() {
    let mut reg = profluens_core::registry::Registry::new();
    profluens_audio::register(&mut reg);
    let desc = reg.get("audiostretch").expect("audiostretch is registered");
    assert_eq!(desc.name, "audiostretch");
    assert!(desc.make_default.is_some(), "constructible from a launch string");
    let rate = desc.props.iter().find(|p| p.name == "rate").expect("rate prop");
    assert!(rate.live, "rate must be settable while playing");

    // The launch-string path, including the decimal spelling `parse` interns as an id.
    let mut p = Pipeline::new();
    reg.parse(&mut p, "audiostretch rate=1.5").expect("parse audiostretch rate=1.5");
}
