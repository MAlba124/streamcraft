//! Musepack properties: the SV8 `SH` stream-header packet, and the SV7 fixed header.
//!
//! In-tree spec: `spec/MPC.md`, written against the Musepack Trac wiki's SV8/SV7 pages (the
//! host is dead; recovered from the Internet Archive and cited rather than vendored, since
//! the pages carry no licence) and the reference reader vendored beside it —
//! `spec/libmpc-streaminfo.c`, `spec/libmpc-streaminfo.h`, `spec/libmpc-mpcdec.h`, BSD-3-Clause
//! (`spec/libmpc-COPYING.txt`). Citations below name the structure and the heading in
//! `spec/MPC.md`.
//!
//! Two unrelated container designs share the extension:
//!
//! * **SV8** — magic `MPCK`, then a flat sequence of `<2 ASCII key><varint size><payload>`
//!   packets. Everything a scan needs is in the first `SH` packet.
//! * **SV7** — magic `MP+` and a version octet, then a fixed 28-octet header of
//!   little-endian words whose fields are bit-packed MSB-first.
//!
//! Tags are an APEv2 block optionally followed by ID3v1, at EOF — [`crate::tail`]'s job,
//! shared with MP3, Monkey's Audio and WavPack. On SV8 they sit behind the `SE` packet.
//!
//! ## What it costs
//!
//! **Two reads**: the prefix for the header, the tail for the tags. Never an extent — the
//! `SH` packet is by definition before the first audio packet. Nothing here allocates.

use crate::Props;

/// SV8's magic (`spec/MPC.md`, "Sniffing").
pub(crate) const MAGIC_SV8: &[u8; 4] = b"MPCK";
/// SV7's magic, followed by a version octet whose low nibble must be 7.
pub(crate) const MAGIC_SV7: &[u8; 3] = b"MP+";
/// The stream version SV7's version octet must carry in its low nibble.
const SV7_VERSION: u8 = 7;
/// The stream version an SV8 `SH` packet must declare.
const SV8_VERSION: u8 = 8;

/// Packets the SV8 walk will visit looking for `SH`. The mandatory ones before the first
/// audio packet are `SH`, `RG` and optionally `EI`/`SO`; a handful covers that.
const MAX_PACKETS: usize = 16;

/// Longest a size varint may be: "n*8; 0 < n < 10" — nine octets, 63 significant bits.
const MAX_VARINT: usize = 9;

/// Samples in one Musepack frame: `MPC_FRAME_LENGTH = 36 * 32` (`spec/libmpc-mpcdec.h`).
const FRAME_LENGTH: u64 = 36 * 32;

/// The synthesis filterbank's latency, removed from a non-gapless SV7 file's sample count:
/// `MPC_DECODER_SYNTH_DELAY = 481` (`spec/libmpc-mpcdec.h`).
const SYNTH_DELAY: u64 = 481;

/// The four sampling frequencies both stream versions index into
/// (`spec/libmpc-streaminfo.c` line 53: `samplefreqs[8] = { 44100, 48000, 37800, 32000 }`).
///
/// The reference array has eight slots and four initialisers, so indices 4–7 are zero and
/// `check_streaminfo` rejects them. SV8's index field is three bits and *can* express them;
/// SV7's is two bits and cannot. Represented here as a four-entry table plus a bounds check,
/// which is the same thing said without the zero-filled trap.
const SAMPLE_RATES: [u32; 4] = [44_100, 48_000, 37_800, 32_000];

/// Read the properties of the Musepack stream beginning at `body` in `window`.
pub(crate) fn props(window: &[u8], body: u64) -> Props {
    let mut props = Props::default();
    let Ok(at) = usize::try_from(body) else { return props };
    let Some(h) = window.get(at..) else { return props };
    if h.starts_with(MAGIC_SV8) {
        sv8(&h[MAGIC_SV8.len()..], &mut props);
    } else if h.starts_with(MAGIC_SV7) {
        sv7(h, &mut props);
    }
    props
}

// --- SV8 ---------------------------------------------------------------------------------

/// Walk the packet chain for the `SH` stream header (`spec/MPC.md`, "Packet framing").
///
/// ```text
///   [ key: 2 ASCII bytes ][ size: varint ][ payload ]
/// ```
///
/// The size **includes the key and the size octets themselves** — "the minimum length of a
/// block is 3 bytes" — so the payload is `size - 2 - size_octets`. Both key characters must
/// be `A`..`Z`, which is the format's own validity test and this walk's stop condition on
/// junk.
fn sv8(mut rest: &[u8], props: &mut Props) {
    for _ in 0..MAX_PACKETS {
        let Some(key) = rest.get(..2) else { return };
        if !key.iter().all(|c| c.is_ascii_uppercase()) {
            return;
        }
        let Some((size, size_len)) = varint(&rest[2..]) else { return };
        let head = 2 + size_len;
        let Ok(size) = usize::try_from(size) else { return };
        if size < head {
            return; // a size that does not even cover its own header
        }
        if key == b"SH" {
            // Clamped to what was actually read: a truncated header yields nothing, not a
            // panic. The payload may also be zero-padded past its fields, which is why the
            // walk advances by `size` and never by the sum of the field widths.
            let end = size.min(rest.len());
            if let Some(payload) = rest.get(head..end) {
                stream_header(payload, props);
            }
            return;
        }
        // `SE` is the last packet of the stream; nothing useful follows it.
        if key == b"SE" {
            return;
        }
        let Some(next) = rest.get(size..) else { return };
        rest = next;
    }
}

/// The `SH` payload (`spec/MPC.md`, "`SH` — the stream header").
///
/// ```text
///   CRC32              32 bits
///   stream version      8 bits   (must be 8)
///   sample count       varint    (a PLAIN value, unlike the packet size)
///   beginning silence  varint    (likewise)
///   sample freq index   3 bits
///   max used bands      5 bits   stored as value - 1
///   channel count       4 bits   stored as value - 1
///   mid/side used       1 bit
///   audio block frames  3 bits
/// ```
///
/// Every field up to the last two octets is byte-aligned — 32 + 8 bits of fixed prologue, then
/// whole-octet varints — so no bit reader is needed until the final 16 bits, which are read as
/// two octets. The CRC is **not** verified: a tag scan reports what a file says about itself
/// and leaves refusing it to a decoder.
fn stream_header(payload: &[u8], props: &mut Props) {
    // Octets 0..4 are the CRC32; the stream version is octet 4.
    let Some(&version) = payload.get(4) else { return };
    if version != SV8_VERSION {
        return;
    }
    let Some((samples, n)) = varint(&payload[5..]) else { return };
    let Some((silence, m)) = varint(&payload[5 + n..]) else { return };
    let Some(packed) = payload.get(5 + n + m..5 + n + m + 2) else { return };

    let rate_index = usize::from(packed[0] >> 5);
    // The `- 1` bias on both counts is the easiest thing to get wrong: a stereo file stores 1.
    let max_band = u32::from(packed[0] & 0x1F) + 1;
    let channels = u32::from(packed[1] >> 4) + 1;

    // `check_streaminfo`'s gate, with the channel ceiling relaxed to the format's own 16 —
    // libmpc caps at 2 because its *decoder* supports no more, which is not a statement about
    // what a file may legally declare (`spec/MPC.md`, "Shared: the sample-frequency table").
    if max_band == 0 || max_band >= 32 || channels == 0 || channels > 16 {
        return;
    }
    let Some(&rate) = SAMPLE_RATES.get(rate_index) else {
        return; // indices 4-7 are undefined; the reference reader fails on them too
    };

    props.sample_rate = Some(rate);
    props.channels = Some(channels);
    // `mpc_streaminfo_get_length`: (samples - beginning silence) / sample frequency. Exact —
    // a declared sample count. A silence longer than the stream is nonsense; saturating
    // rather than wrapping keeps it a zero-length file instead of a 584-year one.
    finish(props, rate, samples.saturating_sub(silence), true);
}

/// A big-endian base-128 varint: seven payload bits per octet in the low bits, the MSB the
/// continuation flag. Returns the value and how many octets it took.
///
/// Only the **packet size** field is self-inclusive; every varint inside a payload — the two
/// here — is a plain value. Bounded at [`MAX_VARINT`] octets, so hostile input cannot make the
/// walk run away.
fn varint(b: &[u8]) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    for i in 0..MAX_VARINT {
        let byte = *b.get(i)?;
        value = value.checked_shl(7)?.checked_add(u64::from(byte & 0x7F))?;
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
    }
    None
}

// --- SV7 ---------------------------------------------------------------------------------

/// The fixed SV7 header (`spec/MPC.md`, "Header — 28 octets").
///
/// SV7 is a stream of 32-bit **little-endian** words that are byte-swapped and then read
/// MSB-first, so each field is a bit range of the little-endian `u32` at its word offset:
///
/// ```text
///   octets 0..3   'M' 'P' '+' version   (low nibble of the version octet must be 7)
///   W1  @4        FrameCount
///   W2  @8        bits 16..17 = sample frequency index
///   W5  @20       bit 31 = TrueGapless, bits 20..30 = LastFrameLength (11 bits)
/// ```
///
/// The channel count is **not stored**: `libmpc-streaminfo.c` sets `si->channels = 2`
/// unconditionally for SV7, so every SV7 file is stereo.
fn sv7(h: &[u8], props: &mut Props) {
    let Some(&version) = h.get(3) else { return };
    if version & 0x0F != SV7_VERSION {
        return;
    }
    let le32 = |at: usize| -> Option<u32> {
        h.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let (Some(frames), Some(w2), Some(w5)) = (le32(4), le32(8), le32(20)) else { return };

    let Some(&rate) = SAMPLE_RATES.get(((w2 >> 16) & 3) as usize) else { return };
    let true_gapless = (w5 >> 31) & 1 == 1;
    let last_frame_samples = u64::from((w5 >> 20) & 0x7FF);

    // `streaminfo_read_header_sv7`, verbatim. The `== 0 → FRAME_LENGTH` normalisation happens
    // *before* the branch, so the non-gapless case always subtracts exactly 481.
    let last = if last_frame_samples == 0 {
        FRAME_LENGTH
    } else if last_frame_samples > FRAME_LENGTH {
        return; // the reference reader fails the file here
    } else {
        last_frame_samples
    };
    let total = u64::from(frames) * FRAME_LENGTH;
    let samples = if true_gapless {
        total.saturating_sub(FRAME_LENGTH - last)
    } else {
        total.saturating_sub(SYNTH_DELAY)
    };

    props.channels = Some(2);
    // TrueGapless means the last frame's real length is in the header, so the count is exact.
    // Without it the tail of the last frame is padding of unknown length and a fixed 481-sample
    // latency stands in — accurate to within one 1152-sample frame, under 30 ms, but an
    // approximation, and `duration_exact` says so.
    finish(props, rate, samples, true_gapless);
}

// --- shared ------------------------------------------------------------------------------

/// `duration = samples / rate`, in nanoseconds.
fn finish(props: &mut Props, rate: u32, samples: u64, exact: bool) {
    props.sample_rate = Some(rate);
    // u128: `samples * 1e9` overflows u64 past ~18 G samples, and an SV8 sample count is a
    // 63-bit varint.
    props.duration_ns = u64::try_from(u128::from(samples) * 1_000_000_000 / u128::from(rate)).ok();
    props.duration_exact = exact;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use crate::fixture::{mpc_sv7, mpc_sv8, mpc_varint};

    #[test]
    fn the_varint_is_big_endian_base_128_with_an_msb_continuation() {
        assert_eq!(varint(&[0x00]), Some((0, 1)));
        assert_eq!(varint(&[0x7F]), Some((127, 1)));
        assert_eq!(varint(&[0x81, 0x00]), Some((128, 2)));
        assert_eq!(varint(&[0xFF, 0x7F]), Some((16_383, 2)));
        // 10_395_840, the sample count in the SV8 specification's own worked example.
        assert_eq!(varint(&mpc_varint(10_395_840)), Some((10_395_840, 4)));
        // Round-trip every power of two the encoding can hold.
        for shift in 0..62 {
            let v = 1u64 << shift;
            assert_eq!(varint(&mpc_varint(v)), Some((v, mpc_varint(v).len())), "1 << {shift}");
        }
        // Truncated, and never-terminating, both yield nothing rather than hanging.
        assert_eq!(varint(&[0x81]), None);
        assert_eq!(varint(&[]), None);
        assert_eq!(varint(&[0xFF; 16]), None, "capped at nine octets");
    }

    #[test]
    fn sv8_stream_header_props_and_exact_duration() {
        // 44100 Hz stereo, 88200 samples with no beginning silence — exactly two seconds.
        let f = mpc_sv8(0, 2, 88_200, 0, &[]);
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(44_100));
        assert_eq!(p.channels, Some(2));
        assert_eq!(p.duration_ns, Some(2_000_000_000));
        assert!(p.duration_exact);
    }

    #[test]
    fn sv8_beginning_silence_is_subtracted() {
        let f = mpc_sv8(1, 1, 48_000 + 1_200, 1_200, &[]);
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(48_000));
        assert_eq!(p.channels, Some(1));
        assert_eq!(p.duration_ns, Some(1_000_000_000), "silence comes off the front");
    }

    #[test]
    fn sv8_every_defined_rate_index_and_undefined_ones_are_rejected() {
        for (index, &rate) in SAMPLE_RATES.iter().enumerate() {
            let f = mpc_sv8(index as u8, 2, u64::from(rate), 0, &[]);
            let p = props(&f, 0);
            assert_eq!(p.sample_rate, Some(rate), "index {index}");
            assert_eq!(p.duration_ns, Some(1_000_000_000));
        }
        for index in 4..8u8 {
            let f = mpc_sv8(index, 2, 44_100, 0, &[]);
            assert_eq!(props(&f, 0), Props::default(), "index {index} is undefined");
        }
    }

    /// The channel and max-band fields are stored biased by one — the single easiest thing to
    /// get wrong, and one that would silently report mono for every stereo file.
    #[test]
    fn the_channel_count_is_stored_biased_by_one() {
        for channels in 1..=16u32 {
            let f = mpc_sv8(0, channels as u8, 44_100, 0, &[]);
            assert_eq!(props(&f, 0).channels, Some(channels), "{channels} channels");
        }
    }

    /// `SH` need not be the first packet; the walk steps over whatever precedes it.
    #[test]
    fn the_walk_finds_sh_behind_other_packets() {
        let f = mpc_sv8(0, 2, 44_100, 0, &[(*b"EI", vec![0x50, 1, 23, 0])]);
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(44_100));
        assert_eq!(p.duration_ns, Some(1_000_000_000));
    }

    #[test]
    fn sv8_junk_and_bad_versions_yield_nothing() {
        // A key outside A-Z ends the walk.
        let mut f = mpc_sv8(0, 2, 44_100, 0, &[]);
        f[4] = b'1';
        assert_eq!(props(&f, 0), Props::default());
        // A stream version that is not 8.
        let mut g = mpc_sv8(0, 2, 44_100, 0, &[]);
        let sh_payload = 4 + 2 + 1; // MPCK + key + one size octet
        g[sh_payload + 4] = 7;
        assert_eq!(props(&g, 0), Props::default());
        assert_eq!(props(b"MPCK", 0), Props::default());
        assert_eq!(props(b"", 0), Props::default());
    }

    #[test]
    fn sv7_true_gapless_is_exact_and_the_old_form_is_not() {
        // 100 frames, last frame 500 samples long, gapless: 99*1152 + 500 samples.
        let f = mpc_sv7(0, 100, true, 500);
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(44_100));
        assert_eq!(p.channels, Some(2), "SV7 is always stereo");
        let samples = 100 * 1152 - (1152 - 500);
        assert_eq!(p.duration_ns, Some(samples * 1_000_000_000 / 44_100));
        assert!(p.duration_exact);

        // Not gapless: a flat 481-sample filterbank latency comes off instead.
        let g = mpc_sv7(0, 100, false, 0);
        let q = props(&g, 0);
        assert_eq!(q.duration_ns, Some((100 * 1152 - 481) * 1_000_000_000 / 44_100));
        assert!(!q.duration_exact, "the last frame's real length is unknown");
    }

    #[test]
    fn sv7_rate_indices_and_bad_versions() {
        for (index, &rate) in SAMPLE_RATES.iter().enumerate() {
            let f = mpc_sv7(index as u8, 100, true, 1_152);
            assert_eq!(props(&f, 0).sample_rate, Some(rate), "index {index}");
        }
        // The version nibble must be 7; the high nibble (PNS) is free.
        let mut f = mpc_sv7(0, 10, true, 1_152);
        f[3] = 0x17;
        assert_eq!(props(&f, 0).sample_rate, Some(44_100), "PNS set is still SV7");
        f[3] = 0x08;
        assert_eq!(props(&f, 0), Props::default());
        // A last-frame length past a whole frame is rejected by the reference reader.
        let g = mpc_sv7(0, 10, true, 1_153);
        assert_eq!(props(&g, 0), Props::default());
    }

    #[test]
    fn truncation_never_panics() {
        for f in [mpc_sv8(0, 2, 44_100, 0, &[(*b"RG", vec![1, 0, 0, 0, 0, 0, 0, 0, 0])]), mpc_sv7(1, 500, true, 900)] {
            for n in 0..f.len() {
                let _ = props(&f[..n], 0);
            }
        }
    }

    #[test]
    fn single_byte_corruption_never_panics() {
        for base in [mpc_sv8(0, 2, 10_395_840, 0, &[]), mpc_sv7(2, 1_000, false, 0)] {
            for i in 0..base.len() {
                for bit in [0x01u8, 0x40, 0x80, 0xFF] {
                    let mut f = base.clone();
                    f[i] ^= bit;
                    let _ = props(&f, 0);
                }
            }
        }
    }
}
