//! `pf_mp3::probe_props` — duration and format from an MP3's first frame, no decoding.
//!
//! Fixtures are hand-built MPEG audio frames: a real header (ISO/IEC 11172-3 §2.4.2.3)
//! followed by the right amount of zeroed side information and, where the test wants one,
//! a Xing/Info or VBRI block at the offset the encoder would have put it. Zeroed payload
//! is exactly what an encoder writes into a VBR-header frame anyway — it carries no audio
//! — so these are the shape of a real file's first 417 bytes, and they let each test pin
//! one variable: side-info width, the CRC word, which VBR layout, or none at all.

// Tests own their fixtures outright; the allocation discipline clippy.toml enforces is
// about `process()` hot paths, not fixture construction.
#![allow(clippy::disallowed_methods)]

use pf_mp3::probe_props;

// Second header byte, up to the bitrate field: 11-bit sync, version, layer, protection.
/// MPEG-1, Layer III, no CRC.
const MPEG1: u8 = 0xFB;
/// MPEG-1, Layer III, **CRC-protected** (the protection bit is active-low).
const MPEG1_CRC: u8 = 0xFA;
/// MPEG-2 (the low-sampling-frequency family), Layer III, no CRC.
const MPEG2: u8 = 0xF3;

/// Channel mode field (§2.4.2.3): `'00'` stereo, `'11'` single_channel.
const STEREO: u8 = 0b00;
const MONO: u8 = 0b11;

/// A 4-byte MPEG audio frame header. `bitrate_index` and `samplerate_index` index the
/// §2.4.2.3 tables for the version in `version_byte`.
fn header(version_byte: u8, bitrate_index: u8, samplerate_index: u8, mode: u8) -> [u8; 4] {
    [0xFF, version_byte, (bitrate_index << 4) | (samplerate_index << 2), mode << 6]
}

/// A whole frame of `len` bytes: the header, zeroed payload, and `tag` planted at
/// `tag_at` bytes from the frame's start.
fn frame(head: [u8; 4], len: usize, tag_at: usize, tag: &[u8]) -> Vec<u8> {
    let mut f = head.to_vec();
    f.resize(len, 0);
    f[tag_at..tag_at + tag.len()].copy_from_slice(tag);
    f
}

/// A Xing VBR header carrying only a frame count (flags bit 0).
fn xing(frames: u32) -> Vec<u8> {
    let mut b = b"Xing".to_vec();
    b.extend_from_slice(&1u32.to_be_bytes());
    b.extend_from_slice(&frames.to_be_bytes());
    b
}

/// The 36-byte LAME extension: a 9-byte encoder version string, zero filler, and at 0x15
/// the three bytes packing the 12-bit encoder delay and 12-bit padding.
fn lame_ext(writer: &[u8], delay: u32, padding: u32) -> [u8; 36] {
    let mut e = [0u8; 36];
    e[..writer.len()].copy_from_slice(writer);
    let v = (delay << 12) | padding;
    e[0x15..0x18].copy_from_slice(&[(v >> 16) as u8, (v >> 8) as u8, v as u8]);
    e
}

/// A Xing header carrying every field LAME writes — frame count, byte count, 100-byte TOC
/// and quality, flags `$F` — with `ext` (a [`lame_ext`], or junk, or nothing) after them.
fn xing_full(frames: u32, ext: &[u8]) -> Vec<u8> {
    let mut b = b"Xing".to_vec();
    b.extend_from_slice(&0xFu32.to_be_bytes());
    b.extend_from_slice(&frames.to_be_bytes());
    b.extend_from_slice(&999_999u32.to_be_bytes()); // byte count
    b.extend_from_slice(&[0u8; 100]); // seek TOC
    b.extend_from_slice(&57u32.to_be_bytes()); // quality
    b.extend_from_slice(ext);
    b
}

/// A Fraunhofer VBRI header: tag, version, delay, quality, byte count, then the frame
/// count at offset 14.
fn vbri(frames: u32) -> Vec<u8> {
    let mut b = b"VBRI".to_vec();
    b.extend_from_slice(&1u16.to_be_bytes());
    b.extend_from_slice(&0u16.to_be_bytes());
    b.extend_from_slice(&0u16.to_be_bytes());
    b.extend_from_slice(&0u32.to_be_bytes());
    b.extend_from_slice(&frames.to_be_bytes());
    b
}

/// `frames × samples_per_frame ÷ rate`, in nanoseconds — the arithmetic the assertions
/// are checking, spelled out independently of the implementation. Widened to `u128`
/// because the product overflows a `u64` well before the frame count field does.
fn duration_ns(frames: u64, samples_per_frame: u64, rate: u64) -> u64 {
    let ns =
        u128::from(frames) * u128::from(samples_per_frame) * 1_000_000_000 / u128::from(rate);
    ns.min(u128::from(u64::MAX)) as u64
}

/// `samples ÷ rate` in nanoseconds — the same arithmetic once a gapless trim has taken the
/// figure off a whole number of frames.
fn samples_ns(samples: u64, rate: u64) -> u64 {
    (u128::from(samples) * 1_000_000_000 / u128::from(rate)) as u64
}

/// MPEG-1 Layer III at 44.1 kHz / 128 kbps: `144 × 128000 ÷ 44100 = 417` bytes.
const MPEG1_128_LEN: usize = 417;
/// MPEG-2 Layer III at 22.05 kHz / 64 kbps: `72 × 64000 ÷ 22050 = 208` bytes.
const MPEG2_64_LEN: usize = 208;

#[test]
fn xing_frame_count_gives_an_exact_duration() {
    // MPEG-1 stereo side info is 32 bytes, so Xing begins at 4 + 32.
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing(100));
    let p = probe_props(&f, MPEG1_128_LEN as u64).expect("first frame must be found");

    assert_eq!(p.sample_rate, 44_100);
    assert_eq!(p.channels, 2);
    assert_eq!(p.bitrate_kbps, Some(128));
    assert!(p.exact, "a frame count is exact, not an estimate");
    // 100 frames × 1152 samples ÷ 44100 Hz = 2.6122448979… s
    assert_eq!(p.duration_ns, Some(duration_ns(100, 1152, 44_100)));
    assert_eq!(p.duration_ns, Some(2_612_244_897));
}

#[test]
fn xing_offset_follows_the_side_info_width() {
    // Mono MPEG-1 side info is 17 bytes, not 32 — Xing moves to 4 + 17. Planting it at
    // the stereo offset instead would leave it unfound, which is the point of the test.
    let f = frame(header(MPEG1, 9, 0, MONO), MPEG1_128_LEN, 21, &xing(20));
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();
    assert_eq!(p.channels, 1);
    assert!(p.exact);
    assert_eq!(p.duration_ns, Some(duration_ns(20, 1152, 44_100)));

    // MPEG-2 (low sampling frequency) stereo: 17 bytes of side info, and 576 samples per
    // frame instead of 1152 (ISO/IEC 13818-3 §2.4.2.1).
    let f = frame(header(MPEG2, 8, 0, STEREO), MPEG2_64_LEN, 21, &xing(50));
    let p = probe_props(&f, MPEG2_64_LEN as u64).unwrap();
    assert_eq!(p.sample_rate, 22_050);
    assert_eq!(p.channels, 2);
    assert_eq!(p.bitrate_kbps, Some(64));
    assert!(p.exact);
    assert_eq!(p.duration_ns, Some(duration_ns(50, 576, 22_050)));
}

#[test]
fn a_crc_protected_first_frame_shifts_the_xing_offset_by_two() {
    // The CRC-check word (§2.4.1) sits between the header and the side info, so Xing lands
    // at 4 + 2 + 32.
    let f = frame(header(MPEG1_CRC, 9, 0, STEREO), MPEG1_128_LEN, 38, &xing(7));
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();
    assert!(p.exact);
    assert_eq!(p.duration_ns, Some(duration_ns(7, 1152, 44_100)));

    // At the un-CRC'd offset the same block is not where the spec says to look, and the
    // probe correctly falls back to the CBR estimate rather than reading a stray count.
    let f = frame(header(MPEG1_CRC, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing(7));
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();
    assert!(!p.exact);
}

#[test]
fn a_lame_extension_trims_the_encoder_delay_and_padding() {
    // LAME's own figures: 576 samples of encoder delay at the head, and enough silence at
    // the tail to fill the last frame. 100 frames code 115200 samples, of which 576 + 1000
    // are the encoder's, leaving 113624 samples of audio.
    let ext = lame_ext(b"LAME3.99r", 576, 1000);
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing_full(100, &ext));
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();

    assert!(p.exact, "the encoder stated both figures; nothing here is estimated");
    assert_eq!(p.duration_ns, Some(samples_ns(100 * 1152 - 576 - 1000, 44_100)));
    assert_eq!(p.duration_ns, Some(2_576_507_936));
    // 1576 samples — 35.7 ms — shorter than the coded length, which is the whole point.
    assert_eq!(duration_ns(100, 1152, 44_100) - p.duration_ns.unwrap(), 35_736_961);

    // The 12-bit fields' full range, on the MPEG-2 side info / 576-sample geometry.
    let ext = lame_ext(b"Lavc60.31", 4095, 4095);
    let f = frame(header(MPEG2, 8, 0, STEREO), MPEG2_64_LEN, 21, &xing_full(1000, &ext));
    let p = probe_props(&f, MPEG2_64_LEN as u64).unwrap();
    assert_eq!(p.duration_ns, Some(samples_ns(1000 * 576 - 8190, 22_050)));
}

#[test]
fn the_lame_extension_sits_after_whatever_the_flags_announce() {
    // Only bit 0 set: no byte count, TOC or quality stand between the frame count and the
    // extension, so it begins 12 bytes into the Xing block instead of 120.
    let mut x = xing(100);
    x.extend_from_slice(&lame_ext(b"Lavf61.7.", 1105, 288));
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &x);
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();
    assert_eq!(p.duration_ns, Some(samples_ns(100 * 1152 - 1105 - 288, 44_100)));

    // The same bytes at the offset the *other* flag set implies are not the extension: a
    // reader that guessed a fixed offset would trim by whatever it found there.
    let mut x = xing(100);
    x.resize(120, 0);
    x.extend_from_slice(&lame_ext(b"Lavf61.7.", 1105, 288));
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &x);
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();
    assert_eq!(p.duration_ns, Some(duration_ns(100, 1152, 44_100)));
}

#[test]
fn a_xing_without_a_usable_lame_extension_is_left_untrimmed() {
    let coded = Some(duration_ns(100, 1152, 44_100));

    // Xing with the fields but no extension behind them — the pre-gapless encoders.
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing_full(100, &[]));
    assert_eq!(probe_props(&f, MPEG1_128_LEN as u64).unwrap().duration_ns, coded);

    // Zeroed, and junk, where the writer string belongs: 24 bits at 0x15 that no known
    // writer put there are not a length, however plausible they look.
    let junk: [&[u8]; 3] =
        [b"\0\0\0\0\0\0\0\0\0", b"Xingified", b"\xff\xff\xff\xff\xff\xff\xff\xff"];
    for writer in junk {
        let ext = lame_ext(writer, 4095, 4095);
        let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing_full(100, &ext));
        let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();
        assert_eq!(p.duration_ns, coded, "trimmed on writer {writer:?}");
        assert!(p.exact);
    }

    // A trim that would consume the whole stream is not a trim: 2 frames are 2304 samples
    // and this claims 4096 + 4095 of them, so the file is reported as coded rather than as
    // an empty (or, unchecked, a wrapped) one.
    let ext = lame_ext(b"LAME3.99r", 4095, 4095);
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing_full(2, &ext));
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();
    assert_eq!(p.duration_ns, Some(duration_ns(2, 1152, 44_100)));

    // Exactly the coded length is the same case, one sample from the boundary: a single
    // frame that is 1000 samples of delay and 152 of padding has no audio in it.
    let ext = lame_ext(b"LAME3.99r", 1000, 152);
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing_full(1, &ext));
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();
    assert_eq!(p.duration_ns, Some(duration_ns(1, 1152, 44_100)));
}

#[test]
fn vbri_frame_count_gives_an_exact_duration() {
    // VBRI ignores the side info and always sits 32 bytes past the 4-byte header. Mono, so
    // its offset (36) is distinct from where a Xing header would be (21).
    let f = frame(header(MPEG1, 9, 0, MONO), MPEG1_128_LEN, 36, &vbri(365));
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();

    assert_eq!(p.channels, 1);
    assert!(p.exact);
    assert_eq!(p.duration_ns, Some(duration_ns(365, 1152, 44_100)));
}

#[test]
fn a_plain_cbr_stream_estimates_from_bitrate_and_byte_count() {
    // Two frames, no VBR header anywhere.
    let mut audio = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 4, &[]);
    audio.extend_from_slice(&frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 4, &[]));
    assert_eq!(audio.len(), 834);

    let p = probe_props(&audio, audio.len() as u64).unwrap();
    assert_eq!(p.sample_rate, 44_100);
    assert_eq!(p.channels, 2);
    assert_eq!(p.bitrate_kbps, Some(128));
    assert!(!p.exact, "a bitrate estimate is not exact");
    // 834 bytes × 8 bits ÷ 128000 bit/s = 52.125 ms
    assert_eq!(p.duration_ns, Some(52_125_000));

    // `audio` is only a window; `audio_len` is the whole (tag-stripped) file, and that is
    // what the estimate scales with.
    let p = probe_props(&audio, 5_000_000).unwrap();
    assert_eq!(p.duration_ns, Some(312_500_000_000)); // 5 MB at 128 kbps ≈ 312.5 s
}

#[test]
fn a_window_too_short_for_the_whole_frame_still_yields_format_and_an_estimate() {
    // The framer reports the header without the frame, so the VBR header inside it cannot
    // be read — but rate, channels and bitrate are all in those first four bytes.
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing(100));
    let p = probe_props(&f[..64], 5_000_000).unwrap();

    assert_eq!(p.sample_rate, 44_100);
    assert_eq!(p.channels, 2);
    assert_eq!(p.bitrate_kbps, Some(128));
    assert!(!p.exact);
    assert_eq!(p.duration_ns, Some(312_500_000_000));
}

#[test]
fn leading_junk_before_the_first_sync_is_skipped() {
    // A tag scanner hands over the first post-tag byte; a stray byte or two before the
    // first syncword must not defeat the probe.
    let mut audio = vec![0x00, 0x11, 0x22, 0xFF, 0x13, 0x44];
    audio.extend_from_slice(&frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing(100)));

    let p = probe_props(&audio, audio.len() as u64).unwrap();
    assert!(p.exact);
    assert_eq!(p.duration_ns, Some(duration_ns(100, 1152, 44_100)));
}

#[test]
fn audio_without_a_frame_sync_probes_to_none() {
    assert!(probe_props(&[], 0).is_none());
    assert!(probe_props(b"not an mpeg stream at all", 25).is_none());
    // A syncword whose header fields are reserved/forbidden is not a frame: bitrate index
    // $F is forbidden and sampling frequency `'11'` is reserved (§2.4.2.3).
    assert!(probe_props(&[0xFF, 0xFB, 0xFC, 0x00, 0x00, 0x00], 6).is_none());
}

#[test]
fn truncated_and_corrupt_windows_never_panic() {
    let lame = xing_full(100, &lame_ext(b"LAME3.99r", 576, 1000));
    let fixtures = [
        frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing(100)),
        frame(header(MPEG1, 9, 0, MONO), MPEG1_128_LEN, 36, &vbri(365)),
        frame(header(MPEG2, 8, 0, STEREO), MPEG2_64_LEN, 21, &xing(50)),
        // The extension is 156 bytes of Xing block, every byte of which the sweep below
        // truncates and corrupts in turn.
        frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &lame),
    ];
    for f in &fixtures {
        for n in 0..=f.len() {
            probe_props(&f[..n], f.len() as u64);
            probe_props(&f[..n], u64::MAX); // the estimate must not overflow either
        }
        // Every single-byte corruption at every offset: a bogus header field or a partly
        // overwritten VBR block must degrade to `None`/an estimate, never to a panic.
        for i in 0..f.len() {
            for byte in [0x00u8, 0x7F, 0x80, 0xFF] {
                let mut corrupt = f.clone();
                corrupt[i] = byte;
                probe_props(&corrupt, f.len() as u64);
            }
        }
    }
    // The largest frame count the field can hold. `frames × 1152 × 10^9` is ~5 × 10^21 —
    // past a `u64` — even though the quotient (~3.5 years of audio) fits comfortably, so
    // the multiply has to be done wide rather than saturated early.
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing(u32::MAX));
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();
    assert_eq!(p.duration_ns, Some(duration_ns(u64::from(u32::MAX), 1152, 44_100)));

    // The estimate saturates instead: a nonsense file length whose nanosecond figure does
    // not fit clamps at `u64::MAX` rather than wrapping to something small.
    let cbr = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 4, &[]);
    assert_eq!(probe_props(&cbr, u64::MAX).unwrap().duration_ns, Some(u64::MAX));
}

// --- The Xing seek TOC ----------------------------------------------------------------
//
// The seek table is the second thing the VBR header exists for: a hundred bytes saying
// "the point i% of the way through the duration is at byte toc[i]/256 of the stream".
// Building one by hand and reading it back is the only way to pin the field walk, because
// the table's offset depends on which *other* optional fields the flags announced.

/// A Xing header with a chosen set of optional fields. `flags` is the raw flags word, and
/// each field is written only if its bit is set — exactly as an encoder writes them, so a
/// test can prove the walk finds the TOC behind a present *or* absent byte-count field.
fn xing_fields(flags: u32, frames: u32, bytes: u32, toc: Option<&[u8; 100]>, ext: &[u8]) -> Vec<u8> {
    let mut b = b"Xing".to_vec();
    b.extend_from_slice(&flags.to_be_bytes());
    if flags & 0x1 != 0 {
        b.extend_from_slice(&frames.to_be_bytes());
    }
    if flags & 0x2 != 0 {
        b.extend_from_slice(&bytes.to_be_bytes());
    }
    if flags & 0x4 != 0 {
        b.extend_from_slice(toc.expect("flags announced a TOC"));
    }
    if flags & 0x8 != 0 {
        b.extend_from_slice(&57u32.to_be_bytes());
    }
    b.extend_from_slice(ext);
    b
}

/// A plausible TOC: `toc[i] ≈ 256 × i / 100`, i.e. a stream whose bytes advance evenly
/// with its time — a CBR file's table, and the one case whose expected values can be
/// written down without reference to any encoder.
fn linear_toc() -> [u8; 100] {
    let mut t = [0u8; 100];
    for (i, e) in t.iter_mut().enumerate() {
        *e = (i * 256 / 100) as u8;
    }
    t
}

#[test]
fn xing_toc_is_read_with_the_byte_count_in_front_of_it() {
    let toc = linear_toc();
    let x = xing_fields(0xF, 1000, 4_000_000, Some(&toc), &lame_ext(b"LAME3.99r", 576, 1000));
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &x);
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();

    assert_eq!(p.toc.expect("flags bit 2 announced a TOC").points, toc);
    assert_eq!(p.stream_bytes, Some(4_000_000), "flags bit 1's byte count");
    // The LAME extension sits *behind* all four fields; finding the trim proves the walk
    // landed on it rather than 100 bytes short.
    assert!(p.exact);
    assert_eq!(p.duration_ns, Some(samples_ns(1000 * 1152 - 1576, 44_100)));
}

#[test]
fn the_toc_walk_follows_the_flags_not_a_fixed_offset() {
    let toc = linear_toc();
    // Frame count + TOC, no byte count and no quality (flags $5): the table now begins
    // four bytes earlier than it did above. A fixed-offset reader would return the wrong
    // hundred bytes here, and there is nothing in the block to tell it so.
    let x = xing_fields(0x5, 500, 0, Some(&toc), &[]);
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &x);
    let p = probe_props(&f, MPEG1_128_LEN as u64).unwrap();
    assert_eq!(p.toc.expect("TOC present").points, toc);
    assert_eq!(p.stream_bytes, None, "no byte count was written");

    // Frame count only (flags $1): no table at all, and the props say so rather than
    // handing back whatever the frame's payload happened to hold.
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing(500));
    assert_eq!(probe_props(&f, MPEG1_128_LEN as u64).unwrap().toc, None);
}

#[test]
fn toc_entries_map_time_hundredths_to_byte_two_hundred_fifty_sixths() {
    let toc = pf_mp3::XingToc { points: linear_toc() };
    const STREAM: u64 = 4_000_000;
    const DURATION: u64 = 200_000_000_000; // 200 s

    // The two ends. `points[0]` is 0, so the first point is the first audio byte.
    assert_eq!(toc.byte_at(0, STREAM), Some(0));
    assert_eq!(toc.time_at(0, DURATION), Some(0));
    // Point 50: 50 % of 200 s, and `toc[50] = 128` → 128/256 of the stream.
    assert_eq!(toc.time_at(50, DURATION), Some(100_000_000_000));
    assert_eq!(toc.byte_at(50, STREAM), Some(STREAM / 2));
    // Point 99, the last: 99 % of the duration, `toc[99] = 253` → 253/256 of the bytes.
    assert_eq!(toc.time_at(99, DURATION), Some(198_000_000_000));
    assert_eq!(toc.byte_at(99, STREAM), Some(253 * STREAM / 256));
    // Past the end of a hundred-point table, both are `None` rather than a wrap or a panic.
    assert_eq!(toc.byte_at(100, STREAM), None);
    assert_eq!(toc.time_at(100, DURATION), None);
    assert_eq!(toc.byte_at(usize::MAX, STREAM), None);
    assert_eq!(toc.time_at(usize::MAX, DURATION), None);
    // A zero-length stream or a zero duration collapses to zero, not a divide-by-zero.
    assert_eq!(toc.byte_at(99, 0), Some(0));
    assert_eq!(toc.time_at(99, 0), Some(0));
    // The widest inputs the types allow: computed in `u128`, so no overflow.
    assert_eq!(pf_mp3::XingToc { points: [255; 100] }.byte_at(0, u64::MAX), Some(u64::MAX / 256 * 255 + 255 * 255 / 256));
    assert!(toc.time_at(99, u64::MAX).is_some());
}

#[test]
fn first_frame_offset_anchors_the_toc_to_the_file() {
    // The TOC's byte fractions are relative to the first frame, so a caller has to know
    // where that is: four bytes of tagger junk here, and the probe reports it.
    let toc = linear_toc();
    let x = xing_fields(0xF, 1000, 4_000_000, Some(&toc), &[]);
    let mut audio = vec![0x00, 0x11, 0x22, 0x33];
    audio.extend_from_slice(&frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &x));

    let p = probe_props(&audio, audio.len() as u64).unwrap();
    assert_eq!(p.first_frame_offset, 4);
    assert!(p.toc.is_some());
    // A window that starts exactly on the syncword reports zero — the ordinary case.
    let clean = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &x);
    assert_eq!(probe_props(&clean, clean.len() as u64).unwrap().first_frame_offset, 0);
}

#[test]
fn a_toc_the_frame_is_too_short_to_hold_is_absent_not_partial() {
    // A 32 kbit/s mono frame is 144 × 32000 ÷ 44100 = 104 bytes, and its Xing block claims
    // all four optional fields — 137 bytes' worth. The table therefore runs past the end of
    // the frame that is supposed to contain it, which is what a corrupt or lying flags word
    // looks like from here. The frame count in front of it is still readable and still
    // exact; the table must be reported *missing*, not padded out of the bytes behind it.
    let x = xing_fields(0xF, 1000, 4_000_000, Some(&linear_toc()), &[]);
    let mut audio = header(MPEG1, 1, 0, MONO).to_vec();
    audio.resize(21, 0); // MPEG-1 mono side info is 17 bytes, so Xing begins at 4 + 17
    audio.extend_from_slice(&x);
    audio.truncate(104); // …but the frame ends here, mid-table
    audio.extend_from_slice(&header(MPEG1, 1, 0, MONO)); // the next syncword confirms it
    audio.resize(208, 0);

    let p = probe_props(&audio, audio.len() as u64).expect("the first frame is confirmed");
    assert_eq!(p.toc, None, "a table the frame is too short to hold is not read");
    assert_eq!(p.stream_bytes, Some(4_000_000), "the fields in front of it still are");
    assert!(p.exact);
    assert_eq!(p.duration_ns, Some(duration_ns(1000, 1152, 44_100)));

    // The same shape one byte at a time: no truncation of the window yields a partial
    // table either — `probe_props` reads the VBR header only from a whole frame.
    let toc = linear_toc();
    let full = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &xing_fields(0xF, 1000, 4_000_000, Some(&toc), &[]));
    for cut in 0..full.len() {
        if let Some(p) = probe_props(&full[..cut], full.len() as u64) {
            assert_eq!(p.toc, None, "a partial window at cut {cut} must not be believed");
        }
    }
    assert_eq!(probe_props(&full, full.len() as u64).unwrap().toc.map(|t| t.points), Some(toc));
}

#[test]
fn toc_reading_never_panics_on_a_corrupt_block() {
    // The truncation/corruption sweep of `truncated_and_corrupt_windows_never_panic`,
    // rerun over a block that carries a table — the 100 extra bytes are 100 more offsets
    // at which a walk could run off the end.
    let x = xing_fields(0xF, 1000, 4_000_000, Some(&linear_toc()), &lame_ext(b"Lavf61.7.100", 1105, 940));
    let f = frame(header(MPEG1, 9, 0, STEREO), MPEG1_128_LEN, 36, &x);
    for n in 0..=f.len() {
        if let Some(p) = probe_props(&f[..n], f.len() as u64) {
            // Whatever comes back, the accessors are total over it.
            if let Some(t) = p.toc {
                for i in 0..=101 {
                    let _ = t.byte_at(i, p.stream_bytes.unwrap_or(0) as u64);
                    let _ = t.time_at(i, p.duration_ns.unwrap_or(0));
                }
            }
        }
    }
    for i in 0..f.len() {
        for byte in [0x00u8, 0x7F, 0x80, 0xFF] {
            let mut corrupt = f.clone();
            corrupt[i] = byte;
            let _ = probe_props(&corrupt, f.len() as u64);
        }
    }
}
