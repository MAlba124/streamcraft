//! WavPack properties: the 32-byte block header, and the metadata sub-blocks that qualify it.
//!
//! In-tree spec: `spec/WavPack5FileFormat.pdf` (and its `.txt` rendering) — *WavPack 4 & 5
//! Binary File / Block Format*, David Bryant, vendored verbatim under WavPack's BSD-3-Clause
//! licence (`spec/WavPack-COPYING.txt`). Section citations below are that document's.
//! `spec/WAVPACK.md` records provenance plus the two facts the document leaves to the
//! reference implementation: the 15-entry sample-rate table and the `ID_SAMPLE_RATE` payload.
//!
//! A WavPack file "consists of a series of WavPack audio blocks" (§1.0), each opening with a
//! self-describing 32-byte header. Everything a tag scan needs is in the first one or two of
//! them, and the tags themselves are an APEv2/ID3v1 pair at EOF (§4.0) — which is
//! [`crate::tail`]'s job, shared with MP3, Monkey's Audio and Musepack.
//!
//! ## What it costs
//!
//! **Two reads.** The prefix covers the block header and its metadata sub-blocks; the tail
//! read covers the APEv2/ID3v1 pair. There is never an extent: no WavPack property lives past
//! the first audio block. Nothing here allocates — the walk is bounds-checked slicing over
//! the read buffer.

use crate::Props;

/// Bytes in a WavPack block header (§2.0).
const HEADER_LEN: usize = 32;

/// The block magic (§2.0: `char ckID [4]; // "wvpk"`).
pub(crate) const MAGIC: &[u8; 4] = b"wvpk";

/// Stream versions this parser trusts. §2.0: "0x402 to 0x410 are valid for decode".
const VERSION_MIN: u16 = 0x402;
const VERSION_MAX: u16 = 0x410;

/// Blocks the walk will visit looking for the first audio block. A file "may contain only
/// metadata [blocks], especially at the beginning and end" (§1.0); a handful covers that, and
/// past it the bytes are not a WavPack file, they are an attack.
const MAX_BLOCKS: usize = 8;

/// Metadata sub-blocks one block's walk will visit (§3.0). A real block carries a dozen.
const MAX_SUBBLOCKS: usize = 64;

/// `total_samples` sentinel: §2.0, "a value of -1 indicates an unknown length" — and the
/// 40-bit form "reserves values with the lower 32 bits all set", *regardless* of the upper
/// eight, so the test is on the low word alone.
const UNKNOWN_LENGTH: u32 = 0xFFFF_FFFF;

/// Flags field bit assignments (§2.0), as masks and shifts.
const MONO_FLAG: u32 = 1 << 2;
const SRATE_LSB: u32 = 23;
const SRATE_MASK: u32 = 0xF << SRATE_LSB;
/// Rate index 15 (`1111`) is "unknown/custom" and sends the parser to `ID_SAMPLE_RATE`.
const SRATE_CUSTOM: u32 = 0xF;

/// Metadata sub-block id bits (§3.0): `0x3F` selects the function, `0x40` says the payload is
/// one byte shorter than its word count, `0x80` says the length field is three bytes.
const ID_FUNCTION: u8 = 0x3F;
const ID_ODD_SIZE: u8 = 0x40;
const ID_LARGE: u8 = 0x80;

/// "contains channel count and channel_mask" (§3.0).
const ID_CHANNEL_INFO: u8 = 0x0D;
/// "non-standard sampling rate info" (§3.0).
const ID_SAMPLE_RATE: u8 = 0x27;

/// The 15 standard sampling rates, indexed by flags bits 26–23 (§2.0: "sampling rate (if one
/// of 15 standard rates)"). The document names the field but not the table; the table is the
/// reference implementation's `src/common_utils.c` line 31, reproduced and cited in
/// `spec/WAVPACK.md`. Index 15 is not in the table — it means "custom", handled separately.
const SAMPLE_RATES: [u32; 15] = [
    6_000, 8_000, 9_600, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
    64_000, 88_200, 96_000, 192_000,
];

/// Read the properties of the WavPack stream beginning at `body` in `window`.
///
/// `window` starts at file offset 0. Everything is derived from the first block that carries
/// audio — §1.0: "the first block that contains audio samples in a WavPack file determines
/// the format of the entire file" — plus `total_samples` from the block whose `block_index`
/// is zero.
pub(crate) fn props(window: &[u8], body: u64) -> Props {
    let mut props = Props::default();
    let Ok(mut at) = usize::try_from(body) else { return props };

    let mut total: Option<u64> = None;
    let mut format_seen = false;

    for _ in 0..MAX_BLOCKS {
        let Some(h) = window.get(at..).and_then(|r| r.get(..HEADER_LEN)) else { break };
        if &h[..4] != MAGIC {
            break;
        }
        let le32 = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        // §2.0: `uint32_t ckSize; // size of entire block (minus 8)`.
        let ck_size = le32(4);
        let version = u16::from_le_bytes([h[8], h[9]]);
        if !(VERSION_MIN..=VERSION_MAX).contains(&version) {
            break;
        }
        let block_index_u8 = h[10];
        let total_samples_u8 = h[11];
        let total_samples = le32(12);
        let block_index = le32(16);
        let block_samples = le32(20);
        let flags = le32(24);

        // The 40-bit `block_index` / `total_samples` accessors, per the `GET_BLOCK_INDEX` /
        // `GET_TOTAL_SAMPLES` macros quoted in `spec/WAVPACK.md`. `block_index` must be zero
        // for `total_samples` to mean anything (§2.0: "this is only valid if block_index == 0").
        let index = u64::from(block_index) | (u64::from(block_index_u8) << 32);
        if total.is_none() && index == 0 && total_samples != UNKNOWN_LENGTH {
            // `- total_samples_u8` is the reserved-value skip the macro applies; it is zero
            // for every file below 2^32 frames, i.e. every file.
            let value = (u64::from(total_samples) + (u64::from(total_samples_u8) << 32))
                .saturating_sub(u64::from(total_samples_u8));
            // A **non-audio** block stating zero states nothing: the reference reader takes a
            // metadata block's count only when it is non-zero (`open_utils.c`:
            // `else if (total_samples == -1 && !GET_BLOCK_INDEX(..) && GET_TOTAL_SAMPLES(..))`
            // — the last term is a truth test). Without that guard a leading metadata block
            // pins the file at zero length and the real count behind it is never read. An
            // *audio* block stating zero is a genuinely empty file and is believed.
            if block_samples != 0 || value != 0 {
                total = Some(value);
            }
        }

        // `block_samples == 0` marks a "non-audio block" (§2.0), which states no format.
        if !format_seen && block_samples != 0 {
            format_seen = true;
            let payload = window
                .get(at + HEADER_LEN..)
                .map(|r| &r[..r.len().min(block_body_len(ck_size))])
                .unwrap_or(&[]);
            let (rate_meta, channels_meta) = sub_blocks(payload);

            let index = (flags & SRATE_MASK) >> SRATE_LSB;
            props.sample_rate = if index == SRATE_CUSTOM {
                // Index 15 says the rate is not one of the standard fifteen; the real value
                // rides an `ID_SAMPLE_RATE` sub-block. Without one there is genuinely no rate
                // in the file, so none is reported (and no duration is derived).
                rate_meta
            } else {
                SAMPLE_RATES.get(index as usize).copied()
            };
            // The header states only mono-or-stereo (§2.0, bit 2). A multichannel file
            // multiplexes stereo and mono blocks and carries the true count in
            // `ID_CHANNEL_INFO` (§1.0, §3.0), which therefore wins when it is present.
            props.channels = channels_meta.or(Some(if flags & MONO_FLAG != 0 { 1 } else { 2 }));
        }

        if format_seen && total.is_some() {
            break;
        }
        // A block is `ckSize + 8` bytes; a zero or absurd size would not advance the walk.
        let Some(next) = usize::try_from(u64::from(ck_size) + 8).ok().and_then(|n| at.checked_add(n))
        else {
            break;
        };
        if next <= at {
            break;
        }
        at = next;
    }

    if let (Some(total), Some(rate)) = (total, props.sample_rate.filter(|r| *r > 0)) {
        // duration = total sample frames / frames per second, in nanoseconds. u128 because
        // `total * 1e9` overflows u64 past ~18 G frames, and `total_samples` is 40 bits.
        // Exact: it is a declared frame count, not a size divided by a bitrate.
        props.duration_ns =
            u64::try_from(u128::from(total) * 1_000_000_000 / u128::from(rate)).ok();
        props.duration_exact = true;
    }
    props
}

/// Bytes of metadata sub-blocks in a block: everything after the 32-byte header to the end of
/// the block (§3.0). `ckSize` covers the block "minus 8", so the body is `ckSize + 8 - 32`.
fn block_body_len(ck_size: u32) -> usize {
    usize::try_from(u64::from(ck_size).saturating_add(8).saturating_sub(HEADER_LEN as u64))
        .unwrap_or(usize::MAX)
}

/// Walk the metadata sub-blocks of one block (§3.0), returning `(sample rate, channels)` for
/// the two that qualify the header's flags.
///
/// ```text
///   uchar id      0x3F function, 0x20 "needn't understand", 0x40 length is one less,
///                 0x80 large block
///   uchar ws      small: data size in 16-bit WORDS
///   uchar ws[3]   large: same, little-endian
///   data[]        padded to an even number of bytes
/// ```
fn sub_blocks(payload: &[u8]) -> (Option<u32>, Option<u32>) {
    let mut rate = None;
    let mut channels = None;
    let mut at = 0usize;

    for _ in 0..MAX_SUBBLOCKS {
        let Some(&id) = payload.get(at) else { break };
        let large = id & ID_LARGE != 0;
        let (words, head) = if large {
            let Some(w) = payload.get(at + 1..at + 4) else { break };
            (u32::from(w[0]) | (u32::from(w[1]) << 8) | (u32::from(w[2]) << 16), 4usize)
        } else {
            let Some(&w) = payload.get(at + 1) else { break };
            (u32::from(w), 2usize)
        };
        // "data, padded to an even number of bytes" — so the stored span is `words * 2`, and
        // `0x40` says the last of those bytes is padding rather than content.
        let stored = usize::try_from(u64::from(words) * 2).unwrap_or(usize::MAX);
        let len = stored - usize::from(id & ID_ODD_SIZE != 0 && stored > 0);
        let Some(data) = payload.get(at + head..).map(|r| &r[..r.len().min(len)]) else { break };

        match id & ID_FUNCTION {
            // 3 or 4 bytes, little-endian; with four the top byte is masked 0x7F — see
            // `spec/WAVPACK.md`, "The `ID_SAMPLE_RATE` (0x27) sub-block payload".
            ID_SAMPLE_RATE if data.len() == 3 || data.len() == 4 => {
                let mut v = u32::from(data[0])
                    | (u32::from(data[1]) << 8)
                    | (u32::from(data[2]) << 16);
                if let Some(&top) = data.get(3) {
                    v |= u32::from(top & 0x7F) << 24;
                }
                rate = Some(v).filter(|&v| v > 0);
            }
            // 1–5 bytes: a channel count then the channel mask. 6 or 7 bytes: the WavPack 5.0
            // "unlimited streams" form, where the count is 12 bits split across two octets and
            // stored biased by one. (Reference: `read_channel_info`, `src/open_utils.c`.)
            ID_CHANNEL_INFO if !data.is_empty() => {
                channels = Some(if data.len() >= 6 {
                    (u32::from(data[0]) | (u32::from(data[2] & 0x0F) << 8)) + 1
                } else {
                    u32::from(data[0])
                })
                .filter(|&c| c > 0);
            }
            _ => {}
        }

        // Total sub-block length is always even and always leaves the walk on an even
        // address, which is what makes the chain self-synchronising (§3.0).
        let Some(next) = at.checked_add(head).and_then(|n| n.checked_add(stored)) else { break };
        if next <= at {
            break;
        }
        at = next;
    }
    (rate, channels)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use crate::fixture::{wavpack, wavpack_block, wv_sub_block};

    #[test]
    fn header_props_and_exact_duration() {
        // 44100 is rate index 9; 88200 frames is exactly two seconds.
        let f = wavpack(44_100, 2, 88_200, &[]);
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(44_100));
        assert_eq!(p.channels, Some(2));
        assert_eq!(p.duration_ns, Some(2_000_000_000));
        assert!(p.duration_exact, "a declared frame count is authoritative");
    }

    #[test]
    fn the_mono_flag_is_the_channel_count_when_nothing_qualifies_it() {
        let p = props(&wavpack(48_000, 1, 24_000, &[]), 0);
        assert_eq!(p.channels, Some(1));
        assert_eq!(p.sample_rate, Some(48_000));
        assert_eq!(p.duration_ns, Some(500_000_000));
    }

    #[test]
    fn every_standard_rate_index_round_trips() {
        for (index, &rate) in SAMPLE_RATES.iter().enumerate() {
            let f = wavpack(rate, 2, rate, &[]);
            let p = props(&f, 0);
            assert_eq!(p.sample_rate, Some(rate), "index {index}");
            assert_eq!(p.duration_ns, Some(1_000_000_000));
        }
    }

    /// Rate index 15 means "custom": the real rate is in an `ID_SAMPLE_RATE` sub-block, and
    /// without one there is no rate in the file at all.
    #[test]
    fn a_custom_rate_comes_from_the_sub_block() {
        let sub = wv_sub_block(ID_SAMPLE_RATE, &37_800u32.to_le_bytes()[..3]);
        let f = wavpack(0, 2, 37_800, &[sub]);
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(37_800));
        assert_eq!(p.duration_ns, Some(1_000_000_000));
        assert!(p.duration_exact);

        // The four-byte form, for rates past 16777215.
        let big = wv_sub_block(ID_SAMPLE_RATE, &20_000_000u32.to_le_bytes());
        let g = wavpack(0, 2, 40_000_000, &[big]);
        assert_eq!(props(&g, 0).sample_rate, Some(20_000_000));

        // No sub-block: index 15 and nothing to resolve it.
        let h = wavpack(0, 2, 1_000, &[]);
        let p = props(&h, 0);
        assert_eq!(p.sample_rate, None);
        assert_eq!(p.duration_ns, None, "no rate means no duration, not a guessed one");
    }

    #[test]
    fn channel_info_beats_the_mono_flag() {
        // The 1–5 byte form: a count then the mask. 6 channels in a "stereo" block.
        let sub = wv_sub_block(ID_CHANNEL_INFO, &[6, 0x3F, 0x00, 0x00]);
        let p = props(&wavpack(48_000, 2, 48_000, &[sub]), 0);
        assert_eq!(p.channels, Some(6));

        // The 6/7-byte WavPack 5.0 form: count is 12 bits, biased by one.
        let wide = wv_sub_block(ID_CHANNEL_INFO, &[7, 3, 0x00, 0x3F, 0x00, 0x00]);
        let p = props(&wavpack(48_000, 2, 48_000, &[wide]), 0);
        assert_eq!(p.channels, Some(8), "(7 | (0 << 8)) + 1");
    }

    #[test]
    fn an_unknown_length_yields_no_duration() {
        let mut f = wavpack(44_100, 2, 44_100, &[]);
        // total_samples at offset 12: the all-ones "unknown" sentinel (§2.0).
        f[12..16].copy_from_slice(&UNKNOWN_LENGTH.to_le_bytes());
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(44_100), "the format is still known");
        assert_eq!(p.channels, Some(2));
        assert_eq!(p.duration_ns, None);
        assert!(!p.duration_exact);
    }

    /// A metadata-only block in front of the audio: `block_samples == 0` states no format, so
    /// the walk must step over it rather than describe the file from it.
    #[test]
    fn a_leading_metadata_block_is_stepped_over() {
        let meta = wavpack_block(0x410, 0, 0, 0, 0, &[]);
        let audio = wavpack(48_000, 1, 96_000, &[]);
        let mut f = meta;
        f.extend_from_slice(&audio);
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(48_000));
        assert_eq!(p.channels, Some(1));
        assert_eq!(p.duration_ns, Some(2_000_000_000));
    }

    #[test]
    fn a_bad_version_or_magic_yields_nothing() {
        let mut f = wavpack(44_100, 2, 44_100, &[]);
        f[8..10].copy_from_slice(&0x0401u16.to_le_bytes()); // below 0x402
        assert_eq!(props(&f, 0), Props::default());
        let mut g = wavpack(44_100, 2, 44_100, &[]);
        g[8..10].copy_from_slice(&0x0411u16.to_le_bytes()); // above 0x410
        assert_eq!(props(&g, 0), Props::default());
        let mut h = wavpack(44_100, 2, 44_100, &[]);
        h[0] = b'W';
        assert_eq!(props(&h, 0), Props::default());
        assert_eq!(props(b"", 0), Props::default());
    }

    #[test]
    fn truncation_never_panics() {
        let f = wavpack(
            44_100,
            2,
            88_200,
            &[wv_sub_block(ID_CHANNEL_INFO, &[2, 3]), wv_sub_block(0x0A, &[0u8; 40])],
        );
        for n in 0..f.len() {
            let _ = props(&f[..n], 0);
        }
    }

    #[test]
    fn single_byte_corruption_never_panics() {
        let base = wavpack(0, 2, 44_100, &[wv_sub_block(ID_SAMPLE_RATE, &[0x44, 0xAC, 0x00])]);
        for i in 0..base.len() {
            for bit in [0x01u8, 0x40, 0x80, 0xFF] {
                let mut f = base.clone();
                f[i] ^= bit;
                let _ = props(&f, 0);
            }
        }
    }

    /// A sub-block whose declared word count runs far past the block must not loop or slice
    /// out of bounds — it simply ends the walk.
    #[test]
    fn an_absurd_sub_block_length_is_bounded() {
        let mut f = wavpack(44_100, 2, 44_100, &[wv_sub_block(ID_CHANNEL_INFO, &[2, 3, 0, 0])]);
        let at = HEADER_LEN;
        f[at] = ID_LARGE | ID_CHANNEL_INFO;
        f[at + 1] = 0xFF;
        f[at + 2] = 0xFF;
        f[at + 3] = 0xFF;
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(44_100), "the header still parsed");
        assert_eq!(p.duration_ns, Some(1_000_000_000));
    }

    /// The other side of the guard above: an *audio* block stating zero is believed, where a
    /// metadata block stating zero is not. Built by hand — the convenience builder cannot
    /// express "carries audio, but the file is zero frames long".
    #[test]
    fn an_audio_block_stating_zero_is_believed() {
        // Rate index 9 = 44100, stereo, first and last block of the sequence.
        let flags = 0x01 | (9 << 23) | (1 << 11) | (1 << 12);
        let f = wavpack_block(0x410, 0, 0, 100, flags, &[]);
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(44_100));
        assert_eq!(p.duration_ns, Some(0));
        assert!(p.duration_exact);
    }
}
