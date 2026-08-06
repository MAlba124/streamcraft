//! Stream duration from the tail of the file: find the last page, read its granule
//! position, convert it to nanoseconds (spec: RFC 3533 §6, field 4; RFC 7845 §4; Vorbis I
//! specification §4.3 / the Ogg Vorbis mapping).
//!
//! Ogg carries no duration field. "The position specified is the total samples encoded after
//! including all packets finished on this page" (RFC 3533 §6) — so the duration of a stream
//! *is* the granule position of its last page, in whatever unit the codec's mapping defines
//! (48 kHz samples for Opus, PCM samples at the identification rate for Vorbis).
//!
//! Finding that page means scanning **backwards** from the end of the file, which is why
//! [`last_granule`] works over a borrowed tail slice: a caller reads the last ~64 KiB — more
//! than the 65307-byte maximum page size (§6), so at least one complete page is always in
//! reach — and hands it here. Nothing here allocates or panics; every candidate is
//! CRC-verified before it is believed.

use crate::page::{PageHeader, CAPTURE_PATTERN, GRANULE_NONE};

/// Opus granule positions are always counted at 48 kHz, regardless of the input or the
/// decoding rate (RFC 7845 §4: "the granule position of an audio data page is ... the total
/// number of 48 kHz PCM samples").
pub const OPUS_GRANULE_RATE: u32 = 48_000;

const NANOS_PER_SEC: u128 = 1_000_000_000;

/// Scan `tail` backwards for the last valid Ogg page and return its granule position (RFC
/// 3533 §6, field 4). `serial`, when given, restricts the search to one logical bitstream —
/// necessary for a multiplexed ("grouped", §4) file, where the last page overall may belong
/// to the video stream rather than the audio one you are timing.
///
/// `tail` is expected to be the last chunk of the file (~64 KiB), but nothing requires that:
/// garbage before, between, or after the pages is skipped, and the scan is safe over any
/// slice, including one cut mid-page at either end.
///
/// ## What counts as "valid"
/// Backwards from the end, every `"OggS"` capture pattern (§6, field 1) is a candidate, and
/// the first candidate that satisfies **all** of the following wins:
///
/// 1. [`PageHeader::parse`] accepts it — which checks `stream_structure_version == 0` (§6,
///    field 2), that the whole declared page (header + segment table + payload) lies inside
///    the slice, and, decisively, **that its CRC-32 matches** (§6, field 7). A `"OggS"` that
///    happens to appear inside another page's payload fails here.
/// 2. It declares at least one segment (§6, field 8). A page with `page_segments == 0`
///    carries no packet data, so no packet can have finished on it and its granule position
///    times nothing; no muxer emits one. Cheap sanity, and it costs nothing real.
///
///    (It is *not* needed to reject a run of zero bytes, a tempting worry given the Ogg CRC
///    has a zero initial value and no final inversion: the checksum covers the capture
///    pattern itself, and `"OggS"` followed by zeros is `OggS_poly · x^184`, which the
///    generator polynomial cannot divide — so such a span always fails rule 1.)
/// 3. Its granule position is not the -1 sentinel (§6, field 4: "no packets finish on this
///    page"), which by definition carries no timing information. This is a deliberate
///    refinement of "the last page's granule": the last page of a stream that ends with a
///    continued packet can be -1, and the answer a caller wants is the last page that
///    actually timestamps something.
/// 4. Its serial matches `serial`, if one was given (§6, field 5).
///
/// ## Policy: a page whose body is not fully inside the slice
/// Never accepted. Verifying the CRC requires the whole page, so a candidate near the end of
/// the slice whose declared payload runs past it (`NeedMoreBody`) is skipped, and the scan
/// continues backwards to an earlier page. Trusting a header alone would mean trusting four
/// bytes of "OggS" plus a zero version byte — the CRC is the only thing that makes a
/// capture-pattern match meaningful. A page cut off by the *start* of the slice has no
/// capture pattern inside it at all and so is never a candidate. In the intended use — a
/// tail read that ends at EOF — the final page is complete in the window, so this costs
/// nothing.
pub fn last_granule(tail: &[u8], serial: Option<u32>) -> Option<u64> {
    if tail.len() < CAPTURE_PATTERN.len() {
        return None;
    }
    for start in (0..=tail.len() - CAPTURE_PATTERN.len()).rev() {
        // Cheap pre-filter on the first byte before the 4-byte compare.
        if tail[start] != CAPTURE_PATTERN[0] || tail[start..start + 4] != CAPTURE_PATTERN {
            continue;
        }
        let Ok(page) = PageHeader::parse(&tail[start..]) else { continue };
        if page.n_segments() == 0 {
            continue;
        }
        let granule = page.granule_position();
        if granule == GRANULE_NONE {
            continue;
        }
        if serial.is_some_and(|s| s != page.serial()) {
            continue;
        }
        return Some(granule);
    }
    None
}

/// Duration of an Opus stream in nanoseconds, from its last granule position and the
/// `pre_skip` of its ID header (RFC 7845 §4 and §5.1).
///
/// Opus granule positions count 48 kHz samples (§4) — always, whatever
/// [`OpusHead::input_sample_rate`](crate::ident::OpusHead::input_sample_rate) says — and the
/// stream begins with `pre_skip` samples of decoder priming that are discarded at playback
/// (§5.1: "the number of samples ... to discard from the decoder output when starting
/// playback"). §4 spells the arithmetic out: "the pre-skip is subtracted from the granule
/// position ... to determine the duration of the stream". Hence
/// `(granule - pre_skip) / 48000`, in nanoseconds.
///
/// Saturating: a granule below `pre_skip` (a stream shorter than its own priming, or a
/// nonsense value) gives 0 rather than wrapping. The intermediate product is computed in
/// `u128`, so no realistic granule can overflow it.
pub fn opus_duration_ns(last_granule: u64, pre_skip: u16) -> u64 {
    let samples = last_granule.saturating_sub(pre_skip as u64);
    let ns = samples as u128 * NANOS_PER_SEC / OPUS_GRANULE_RATE as u128;
    ns.min(u64::MAX as u128) as u64
}

/// Duration of a Vorbis stream in nanoseconds, from its last granule position and the sample
/// rate in its identification header (Vorbis I §4.2.2 for the rate; the Ogg Vorbis mapping
/// for the granule: a Vorbis page's granule position is the count of PCM samples decodable
/// up to and including that page, at `sample_rate`).
///
/// `None` when `sample_rate` is 0 — which the identification header forbids (§4.2.2), so
/// [`parse_vorbis_ident`](crate::ident::parse_vorbis_ident) can never hand one over; the
/// `Option` exists so a caller passing a rate from elsewhere cannot divide by zero.
///
/// Unlike Opus there is no pre-skip: Vorbis expresses the same idea by *starting* the stream
/// at a granule position that already accounts for the priming, so the granule is the sample
/// count directly.
pub fn vorbis_duration_ns(last_granule: u64, sample_rate: u32) -> Option<u64> {
    if sample_rate == 0 {
        return None;
    }
    let ns = last_granule as u128 * NANOS_PER_SEC / sample_rate as u128;
    Some(ns.min(u64::MAX as u128) as u64)
}

#[cfg(test)]
// Tests build page fixtures; the scan under test allocates nothing (spec: allocation
// discipline — tests are the sanctioned exception).
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::page::{flags, write_page};

    /// Bytes that are not a page: a run of plausible-but-wrong junk.
    fn garbage(n: usize, seed: u8) -> Vec<u8> {
        (0..n).map(|i| seed.wrapping_add(i as u8).wrapping_mul(31) | 1).collect()
    }

    /// A page whose payload *contains* the capture pattern, followed by non-zero bytes so
    /// the fake candidate has version 0 (passing the cheapest check) and must be rejected by
    /// its CRC rather than by luck.
    fn payload_with_fake_capture() -> Vec<u8> {
        let mut p = b"OggS\x00\x00".to_vec();
        p.extend_from_slice(&garbage(60, 7));
        p
    }

    #[test]
    fn finds_the_last_page_in_a_noisy_tail() {
        // A partial leading page (a real page cut at the front), two valid pages — the first
        // of which hides an "OggS" in its payload — then trailing garbage.
        let mut lead = Vec::new();
        write_page(&mut lead, 0, 1_000, 0xAAAA, 1, &[10], &garbage(10, 3));
        let mut tail = lead[13..].to_vec(); // cut mid-header: no capture pattern survives

        let body = payload_with_fake_capture();
        write_page(&mut tail, 0, 2_000, 0xAAAA, 2, &[body.len() as u8], &body);
        write_page(&mut tail, flags::EOS, 3_000, 0xAAAA, 3, &[4], b"last");
        tail.extend_from_slice(&garbage(37, 11));

        assert_eq!(last_granule(&tail, None), Some(3_000));
        assert_eq!(last_granule(&tail, Some(0xAAAA)), Some(3_000));
    }

    #[test]
    fn a_fake_capture_pattern_in_a_payload_is_rejected_by_the_crc() {
        // Only one real page, and its payload contains "OggS\0..." — the scan must return
        // the real page's granule, not stop at the impostor inside it.
        let body = payload_with_fake_capture();
        let mut tail = Vec::new();
        write_page(&mut tail, 0, 4_242, 7, 0, &[body.len() as u8], &body);
        assert_eq!(last_granule(&tail, None), Some(4_242));
    }

    #[test]
    fn a_run_of_zero_bytes_is_not_a_page() {
        // "OggS" + zeros passes the capture-pattern and version checks and *looks* like an
        // empty page; the CRC is what rejects it (rule 1).
        let mut tail = Vec::new();
        write_page(&mut tail, 0, 5_000, 1, 0, &[4], b"real");
        tail.extend_from_slice(b"OggS");
        tail.extend_from_slice(&[0u8; 32]);
        assert!(PageHeader::parse(&tail[tail.len() - 36..]).is_err());
        assert_eq!(last_granule(&tail, None), Some(5_000));
    }

    #[test]
    fn a_zero_segment_page_is_skipped() {
        // A CRC-valid page declaring no segments (rule 2): it carries no packet data, so no
        // packet finished on it and its granule times nothing — the previous page answers.
        let mut tail = Vec::new();
        write_page(&mut tail, 0, 6_000, 1, 0, &[4], b"real");
        write_page(&mut tail, 0, 7_777, 1, 1, &[], &[]);
        assert!(PageHeader::parse(&tail[tail.len() - 27..]).is_ok(), "the impostor is CRC-valid");
        assert_eq!(last_granule(&tail, None), Some(6_000));
    }

    #[test]
    fn skips_pages_with_no_finishing_packet() {
        // The last page declares -1 ("no packets finish on this page", §6) — the answer is
        // the last page that actually timestamps something.
        let mut tail = Vec::new();
        write_page(&mut tail, 0, 9_000, 1, 0, &[4], b"aaaa");
        write_page(&mut tail, 0, GRANULE_NONE, 1, 1, &[4], b"bbbb");
        assert_eq!(last_granule(&tail, None), Some(9_000));
    }

    #[test]
    fn serial_filtering_picks_the_right_logical_stream() {
        // Two grouped streams (§4); the video-ish one ends last.
        let mut tail = Vec::new();
        write_page(&mut tail, 0, 48_000, 0xA1, 5, &[4], b"aud1");
        write_page(&mut tail, 0, 90_000, 0xB2, 5, &[4], b"vid1");
        write_page(&mut tail, 0, 96_000, 0xA1, 6, &[4], b"aud2");
        write_page(&mut tail, flags::EOS, 180_000, 0xB2, 6, &[4], b"vid2");

        assert_eq!(last_granule(&tail, None), Some(180_000));
        assert_eq!(last_granule(&tail, Some(0xA1)), Some(96_000));
        assert_eq!(last_granule(&tail, Some(0xB2)), Some(180_000));
        assert_eq!(last_granule(&tail, Some(0xC3)), None); // no such stream
    }

    #[test]
    fn a_page_whose_body_runs_past_the_slice_is_not_trusted() {
        let mut whole = Vec::new();
        write_page(&mut whole, 0, 100, 1, 0, &[4], b"aaaa");
        let complete = whole.len();
        write_page(&mut whole, 0, 200, 1, 1, &[8], b"bbbbbbbb");

        // Cut inside the second page: its CRC is unverifiable, so the first page answers.
        for cut in complete + 1..whole.len() {
            assert_eq!(last_granule(&whole[..cut], None), Some(100), "cut at {cut}");
        }
        assert_eq!(last_granule(&whole, None), Some(200));
    }

    #[test]
    fn truncations_and_junk_never_panic() {
        let mut tail = Vec::new();
        write_page(&mut tail, 0, 1, 1, 0, &[4], b"aaaa");
        write_page(&mut tail, 0, 2, 1, 1, &[4], b"bbbb");
        for n in 0..tail.len() {
            let _ = last_granule(&tail[..n], None);
            let _ = last_granule(&tail[n..], Some(1));
        }
        assert_eq!(last_granule(&[], None), None);
        assert_eq!(last_granule(b"Ogg", None), None);
        assert_eq!(last_granule(&garbage(1_000, 5), None), None);
        assert_eq!(last_granule(&[b'O'; 500], None), None);
    }

    #[test]
    fn opus_duration_subtracts_pre_skip_at_48khz() {
        // 10 seconds of audio plus the usual 312-sample (6.5 ms) pre-skip.
        assert_eq!(opus_duration_ns(48_000 * 10 + 312, 312), 10_000_000_000);
        // Exactly one 20 ms Opus frame after the pre-skip.
        assert_eq!(opus_duration_ns(960 + 312, 312), 20_000_000);
        // A granule inside the priming, and a zero granule: 0, not a wrap.
        assert_eq!(opus_duration_ns(100, 312), 0);
        assert_eq!(opus_duration_ns(0, 0), 0);
        // The input sample rate never enters the arithmetic — a 44.1 kHz-sourced stream is
        // still counted at 48 kHz (§4).
        assert_eq!(opus_duration_ns(48_000, 0), 1_000_000_000);
        // No overflow on an absurd granule (u128 intermediate).
        assert!(opus_duration_ns(u64::MAX / 2, 0) > 0);
    }

    #[test]
    fn vorbis_duration_uses_the_ident_rate() {
        assert_eq!(vorbis_duration_ns(44_100, 44_100), Some(1_000_000_000));
        assert_eq!(vorbis_duration_ns(44_100 * 191 + 22_050, 44_100), Some(191_500_000_000));
        assert_eq!(vorbis_duration_ns(0, 48_000), Some(0));
        assert_eq!(vorbis_duration_ns(1, 44_100), Some(22_675)); // truncating, not rounding
        assert_eq!(vorbis_duration_ns(1_000, 0), None);
        assert!(vorbis_duration_ns(u64::MAX / 2, 8_000).is_some());
    }

    /// End to end: an Opus file's bos page carries `OpusHead`, its last page carries the
    /// granule — put the two together and you have the duration, with no decoder involved.
    #[test]
    fn opus_head_plus_last_granule_gives_the_duration() {
        use crate::ident::parse_opus_head;

        const PRE_SKIP: u16 = 356;
        let mut head = b"OpusHead".to_vec();
        head.push(1); // version
        head.push(2); // channels
        head.extend_from_slice(&PRE_SKIP.to_le_bytes());
        head.extend_from_slice(&44_100u32.to_le_bytes()); // original input rate
        head.extend_from_slice(&0i16.to_le_bytes()); // output gain
        head.push(0); // mapping family

        // 3 minutes 41 seconds of audio, at Opus's fixed 48 kHz granule rate, plus pre-skip.
        let want_ns = 221_000_000_000u64;
        let granule = 48_000 * 221 + PRE_SKIP as u64;

        let mut file = Vec::new();
        write_page(&mut file, flags::BOS, 0, 0x1234, 0, &[head.len() as u8], &head);
        write_page(&mut file, 0, 0, 0x1234, 1, &[4], b"tags");
        write_page(&mut file, flags::EOS, granule, 0x1234, 2, &[4], b"audi");

        // The head is the bos page's payload; the granule comes from the tail scan.
        let bos = PageHeader::parse(&file).expect("bos page");
        let head = parse_opus_head(bos.payload()).expect("OpusHead");
        assert_eq!(head.pre_skip, PRE_SKIP);
        assert_eq!(head.input_sample_rate, 44_100);

        let last = last_granule(&file, Some(0x1234)).expect("last granule");
        assert_eq!(last, granule);
        assert_eq!(opus_duration_ns(last, head.pre_skip), want_ns);
    }
}
