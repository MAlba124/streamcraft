//! Seek and mid-stream format changes across the three `audio/raw` glue elements —
//! `audioconvert`, `audioresample` and `audiodownmix` (spec: flush/seek; Formats — dynamic
//! caps). All three share one shape and therefore shared one pair of bugs, so they are tested
//! together here rather than three times over in three files.
//!
//! **Flush.** Each element holds back a partial interchannel frame in a `carry` so that only
//! whole frames reach its DSP. That carry is *pre-seek* audio. Surviving a seek, it gets
//! prepended to the first post-seek buffer, and from then on every sample sits a part-frame
//! late in the interleave — on a stereo stream a permanent L/R swap, on 5.1 a channel rotation.
//! It never resyncs, because the offset is carried forward rather than corrected.
//! `audioresample` holds a second kind of pre-seek audio: the per-channel FIR delay lines,
//! which a streaming resampler convolves across buffer boundaries by design, so without a reset
//! the first filter-length of post-seek output is a blend of both sides of the seek.
//!
//! **Relatch.** All three latch the input format at the *first* negotiation and short-circuit
//! ever after, so a later `FormatChange` is silently ignored and every following buffer is read
//! at the wrong width, stride or channel count. Following the change is gapless-playback work
//! with its own design; what these tests pin down is the contract in the meantime — keep the
//! old format, but post a `BusMessage::Warning` saying so, exactly once per distinct change.

// Test setup, not an element hot path: signal generation and byte staging allocate freely (the
// sanctioned whole-file exception in `clippy.toml`).
#![allow(clippy::disallowed_methods)]

use profluens_audio::format::{FIELD_CHANNELS, FIELD_RATE, FIELD_SAMPLE};
use profluens_audio::{
    downmix_to_stereo, AudioConvert, AudioDownmix, AudioResample, SampleFormat,
};
use profluens_core::bus::BusMessage;
use profluens_core::element::Element;
use profluens_core::event::Event;
use profluens_core::format::ValueDesc;
use profluens_core::harness::Harness;

// --- Helpers --------------------------------------------------------------------------

fn f32_bytes(input: &[f32]) -> Vec<u8> {
    input.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn take_f32(out: &mut Vec<f32>, bytes: &[u8]) {
    for w in bytes.as_chunks::<4>().0 {
        out.push(f32::from_le_bytes(*w));
    }
}

fn caps(rate: i64, channels: i64, sample: &'static str) -> [(&'static str, ValueDesc); 3] {
    [
        (FIELD_RATE, ValueDesc::Int(rate)),
        (FIELD_CHANNELS, ValueDesc::Int(channels)),
        (FIELD_SAMPLE, ValueDesc::Id(sample)),
    ]
}

fn rig(element: impl Element + 'static, rate: i64, channels: i64, sample: &'static str) -> Harness {
    let mut h = Harness::with_slot_size(element, 1 << 16);
    h.fix_format("sink", "audio/raw", &caps(rate, channels, sample));
    h.start().expect("start");
    h
}

/// Install new negotiated caps on the sink and build the matching `FormatChange` — the two
/// halves of what the scheduler does when an upstream re-announces at runtime.
fn format_change(h: &mut Harness, rate: i64, channels: i64, sample: &'static str) -> Event {
    let fields = caps(rate, channels, sample);
    h.fix_format("sink", "audio/raw", &fields);
    let fixed = h
        .vocabulary()
        .build_fixed("audio/raw", &fields)
        .expect("the harness vocabulary interned these names in fix_format");
    Event::FormatChange(fixed)
}

fn warnings(h: &mut Harness) -> usize {
    h.bus_messages()
        .iter()
        .filter(|m| matches!(m, BusMessage::Warning { .. }))
        .count()
}

/// Push `bytes` in one buffer and collect everything the element emits.
fn push_all(h: &mut Harness, bytes: &[u8]) -> Vec<u8> {
    let buf = h.alloc(bytes);
    h.push("sink", buf).expect("push");
    let mut out = Vec::new();
    while let Some(b) = h.pull("src") {
        out.extend_from_slice(b.memory.data());
    }
    out
}

// --- audioconvert ---------------------------------------------------------------------

#[test]
fn audioconvert_flush_drops_the_partial_frame_carry() {
    // Stereo S16 → S32. A 6-byte buffer is one whole 4-byte frame plus half of the next, so 2
    // bytes are parked in the carry. Post-seek, L = +1000 and R = -1000 in every frame, so a
    // half-frame shift is unmistakable: the lanes swap sign.
    let mut h = rig(AudioConvert::new(SampleFormat::S32), 48_000, 2, "s16");
    let stale = push_all(&mut h, &[0x11u8, 0x11, 0x22, 0x22, 0x33, 0x33]);
    assert_eq!(stale.len(), 8, "one whole frame converted; 2 bytes held in the carry");

    h.push_event(Event::FlushStart).expect("flush");

    let frames = 1_024usize;
    let pcm: Vec<u8> = (0..frames)
        .flat_map(|_| [1_000i16, -1_000])
        .flat_map(|s| s.to_le_bytes())
        .collect();
    let got = push_all(&mut h, &pcm);

    let want: Vec<u8> = (0..frames)
        .flat_map(|_| [1_000i32 << 16, (-1_000i32) << 16])
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(
        got.len(),
        want.len(),
        "post-flush output is the wrong length — the stale carry was prepended"
    );
    assert_eq!(got, want, "the carry survived the flush and shifted the interleave");
}

#[test]
fn audioconvert_reports_an_ignored_format_change_once() {
    let mut h = rig(AudioConvert::new(SampleFormat::S32), 48_000, 2, "s16");
    assert_eq!(warnings(&mut h), 0, "nothing to report before any change");

    let ev = format_change(&mut h, 44_100, 2, "f32");
    h.push_event(ev).expect("format change");
    assert_eq!(warnings(&mut h), 1, "a mid-stream format change must be reported");

    // The format really is *ignored*: 64 bytes are still read as the latched S16 (16 stereo
    // frames → 128 bytes of S32). Had it relatched to F32 the same bytes would be 8 frames and
    // the output would be 64 bytes.
    let got = push_all(&mut h, &[0u8; 64]);
    assert_eq!(got.len(), 128, "the element followed the new format instead of keeping the old");

    // Repeating the same change, and streaming through it, says nothing more — once per
    // change, never per buffer.
    let ev = format_change(&mut h, 44_100, 2, "f32");
    h.push_event(ev).expect("format change");
    for _ in 0..4 {
        push_all(&mut h, &[0u8; 64]);
    }
    assert_eq!(warnings(&mut h), 0, "the warning repeated");

    // A genuinely different format earns its own warning.
    let ev = format_change(&mut h, 96_000, 6, "s16");
    h.push_event(ev).expect("format change");
    assert_eq!(warnings(&mut h), 1, "a new distinct format must warn again");
}

// --- audioresample --------------------------------------------------------------------

#[test]
fn audioresample_flush_clears_the_filter_delay_line() {
    // Pre-seek is full-scale DC, post-seek is digital silence. A surviving delay line would
    // convolve that DC into the silence for one filter length — an audible smear at exactly
    // the point the listener asked for a clean cut. With the reset the output is *identically*
    // zero, because every tap sees a zero.
    let mut h = rig(AudioResample::new(24_000), 48_000, 1, "f32");
    push_all(&mut h, &f32_bytes(&vec![1.0f32; 8_192]));

    h.push_event(Event::FlushStart).expect("flush");

    let got = push_all(&mut h, &f32_bytes(&vec![0.0f32; 8_192]));
    let mut out = Vec::new();
    take_f32(&mut out, &got);
    assert!(!out.is_empty(), "expected output after the flush");
    for (i, s) in out.iter().enumerate() {
        assert_eq!(*s, 0.0, "sample {i} = {s}: pre-seek audio smeared through the delay line");
    }
}

#[test]
fn audioresample_flush_drops_the_partial_frame_carry() {
    // Stereo F32 (8-byte stride): a 12-byte buffer leaves 4 bytes — half a frame — in the
    // carry. Post-seek the two lanes carry opposite signs, so a shift shows up as a sign flip.
    let mut h = rig(AudioResample::new(24_000), 48_000, 2, "f32");
    push_all(&mut h, &[0xAAu8; 12]);

    h.push_event(Event::FlushStart).expect("flush");

    let frames = 8_192usize;
    let input: Vec<f32> = (0..frames).flat_map(|_| [1.0f32, -1.0]).collect();
    let got = push_all(&mut h, &f32_bytes(&input));
    let mut out = Vec::new();
    take_f32(&mut out, &got);
    assert!(out.len() >= 2, "expected resampled output");
    assert_eq!(out.len() % 2, 0, "output is not a whole number of stereo frames");
    // Halving the rate of a constant-per-lane signal gives back that same constant per lane
    // (the filter is DC-normalised), so the lanes are directly readable. The leading samples
    // are the filter's warm-up transient, so check the settled body.
    for (i, frame) in out.as_chunks::<2>().0.iter().enumerate().skip(64) {
        assert!(
            frame[0] > 0.0 && frame[1] < 0.0,
            "frame {i} = {frame:?}: the carry survived the flush and swapped the lanes"
        );
    }
}

#[test]
fn audioresample_reports_an_ignored_format_change_once() {
    let mut h = rig(AudioResample::new(24_000), 48_000, 1, "f32");
    assert_eq!(warnings(&mut h), 0, "nothing to report before any change");

    let ev = format_change(&mut h, 44_100, 2, "s16");
    h.push_event(ev).expect("format change");
    assert_eq!(warnings(&mut h), 1, "a mid-stream format change must be reported");

    let ev = format_change(&mut h, 44_100, 2, "s16");
    h.push_event(ev).expect("format change");
    for _ in 0..4 {
        push_all(&mut h, &f32_bytes(&vec![0.0f32; 256]));
    }
    assert_eq!(warnings(&mut h), 0, "the warning repeated");

    let ev = format_change(&mut h, 96_000, 6, "s32");
    h.push_event(ev).expect("format change");
    assert_eq!(warnings(&mut h), 1, "a new distinct format must warn again");
}

// --- audiodownmix ---------------------------------------------------------------------

/// A 5.1 S16 test pattern: every channel carries a distinct constant, so a lane shift changes
/// which channel lands in which matrix position and moves the fold's output.
fn surround_pcm(frames: usize) -> Vec<u8> {
    (0..frames)
        .flat_map(|_| [1_000i16, -1_000, 500, 200, 300, -300])
        .flat_map(|s| s.to_le_bytes())
        .collect()
}

#[test]
fn audiodownmix_flush_drops_the_partial_frame_carry() {
    // 6-channel S16 is a 12-byte stride; a 14-byte buffer leaves 2 bytes in the carry.
    let mut h = rig(AudioDownmix::new(), 48_000, 6, "s16");
    push_all(&mut h, &[0x55u8; 14]);

    h.push_event(Event::FlushStart).expect("flush");

    let frames = 1_024usize;
    let pcm = surround_pcm(frames);
    let got = push_all(&mut h, &pcm);

    // Ground truth from the pure library (independently unit-tested).
    let mut want = vec![0u8; frames * 2 * 2];
    downmix_to_stereo(SampleFormat::S16, 6, &pcm, &mut want).expect("fold");
    assert_eq!(
        got.len(),
        want.len(),
        "post-flush output is the wrong length — the stale carry was prepended"
    );
    assert_eq!(got, want, "the carry survived the flush and rotated the channels");
}

#[test]
fn audiodownmix_reports_an_ignored_format_change_once() {
    let mut h = rig(AudioDownmix::new(), 48_000, 6, "s16");
    assert_eq!(warnings(&mut h), 0, "nothing to report before any change");

    // Channel count is the field that matters most here: the fold matrix is chosen for it.
    let ev = format_change(&mut h, 48_000, 8, "s16");
    h.push_event(ev).expect("format change");
    assert_eq!(warnings(&mut h), 1, "a mid-stream channel-count change must be reported");

    // Still folding as 6-channel: 12 bytes in (one latched frame) is 4 bytes of stereo out.
    let got = push_all(&mut h, &surround_pcm(1));
    assert_eq!(got.len(), 4, "the element followed the new channel count instead of the old");

    let ev = format_change(&mut h, 48_000, 8, "s16");
    h.push_event(ev).expect("format change");
    for _ in 0..4 {
        push_all(&mut h, &surround_pcm(4));
    }
    assert_eq!(warnings(&mut h), 0, "the warning repeated");

    let ev = format_change(&mut h, 44_100, 2, "f32");
    h.push_event(ev).expect("format change");
    assert_eq!(warnings(&mut h), 1, "a new distinct format must warn again");
}

// --- The shared invariant -------------------------------------------------------------

#[test]
fn a_flush_never_disturbs_the_learned_format() {
    // A seek moves the read head; it does not change what the upstream is sending. If a flush
    // dropped the latched format the element would have to re-infer it, and a stream whose
    // caps were only ever fixed at link time would stall or error on the next buffer.
    let mut h = rig(AudioConvert::new(SampleFormat::S32), 48_000, 2, "s16");
    for _ in 0..5 {
        h.push_event(Event::FlushStart).expect("flush");
        // 4 bytes = one latched S16 stereo frame → 8 bytes of S32.
        let got = push_all(&mut h, &[0u8; 4]);
        assert_eq!(got.len(), 8, "the flush dropped the latched input format");
    }
    assert_eq!(warnings(&mut h), 0, "a plain flush is not a format change");
}
