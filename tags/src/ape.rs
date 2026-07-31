//! Monkey's Audio (`.ape`) properties: the descriptor/header pair modern files open with, and
//! the single combined header of files from before 3.98.
//!
//! In-tree spec: `spec/APE.md`, written from the Monkey's Audio SDK headers vendored beside it
//! (`spec/MonkeysAudio-APEHeader.h`, `spec/MonkeysAudio-MACLib.h`) under the SDK's
//! three-clause BSD licence (`spec/MonkeysAudio-LICENSE.txt`). Citations below name the
//! structure and the heading in `spec/APE.md`.
//!
//! Tags are an APEv2 block — the format Monkey's Audio *originated*, which WavPack and
//! Musepack then adopted — optionally followed by ID3v1, both at EOF. That is
//! [`crate::tail`]'s job, shared with MP3, WavPack and Musepack.
//!
//! ## What it costs
//!
//! **Two reads.** The prefix covers the header (52 + 24 octets at the very front of the file,
//! or 32 for an old one); the tail read covers the APEv2/ID3v1 pair. There is never an
//! extent. Nothing here allocates.
//!
//! ## Where the header is
//!
//! The SDK scans up to a megabyte for the magic, having first skipped a leading ID3v2 tag.
//! This parser looks only at offset 0 and immediately behind an ID3v2 tag — the two places
//! [`crate::sniff`] can identify without reading more of the file. A `.ape` with arbitrary
//! junk in front of the magic is reported as [`Format::Unknown`](crate::Format), which is
//! honest, rather than being hunted for at the cost of every other file's scan.

use crate::Props;

/// The file magic (`APE_COMMON_HEADER::cID`): `'MAC '`. MAC 12.x also writes `'MACF'` for
/// large files; the layout is identical.
pub(crate) const MAGIC: &[u8; 4] = b"MAC ";
pub(crate) const MAGIC_LARGE: &[u8; 4] = b"MACF";

/// `nVersion` is the version times 1000 (3.99 → 3990). At and above this, a file opens with
/// `APE_DESCRIPTOR` + `APE_HEADER`; below it, with the single `APE_HEADER_OLD`.
const NEW_FORMAT: u16 = 3980;

/// `sizeof(APE_DESCRIPTOR)` and `sizeof(APE_HEADER)` as the current SDK declares them. Both
/// are floors rather than fixed strides — the descriptor states its own size and the header's
/// so that a later SDK may grow either (`spec/APE.md`, "Version ≥ 3980").
const DESCRIPTOR_LEN: usize = 52;
const HEADER_LEN: usize = 24;

/// `sizeof(APE_HEADER_OLD)`.
const OLD_HEADER_LEN: usize = 32;

/// Compression level at which the SDK relaxes its blocks-per-frame ceiling
/// (`APE_COMPRESSION_LEVEL_INSANE`).
const LEVEL_INSANE: u16 = 5000;

/// The SDK's own sanity bounds on `nBlocksPerFrame`, and on the channel count
/// (`APE_MINIMUM_CHANNELS` / `APE_MAXIMUM_CHANNELS`). Reproduced so a corrupt header yields
/// no duration rather than a plausible-looking wrong one.
const MAX_BLOCKS_PER_FRAME: u32 = 1_000_000;
const MAX_BLOCKS_PER_FRAME_INSANE: u32 = 10_000_000;
const MAX_CHANNELS: u16 = 32;

/// Format flag bits this parser reads (`spec/APE.md`, "Format flags"). Only the two sample
/// widths matter here, and only for old files, where `nBitsPerSample` is not stored.
const FLAG_8_BIT: u16 = 1;
const FLAG_24_BIT: u16 = 8;

/// Read the properties of the Monkey's Audio stream beginning at `body` in `window`.
pub(crate) fn props(window: &[u8], body: u64) -> Props {
    let mut props = Props::default();
    let Ok(at) = usize::try_from(body) else { return props };
    let Some(h) = window.get(at..) else { return props };
    if !h.starts_with(MAGIC) && !h.starts_with(MAGIC_LARGE) {
        return props;
    }
    let Some(v) = h.get(4..6) else { return props };
    let version = u16::from_le_bytes([v[0], v[1]]);
    if version >= NEW_FORMAT {
        modern(h, &mut props);
    } else {
        legacy(h, version, &mut props);
    }
    props
}

/// Little-endian field accessors over a header slice.
fn le16(h: &[u8], at: usize) -> Option<u16> {
    h.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]))
}
fn le32(h: &[u8], at: usize) -> Option<u32> {
    h.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Version ≥ 3980: `APE_DESCRIPTOR` then `APE_HEADER` (`spec/APE.md`, "Version ≥ 3980").
///
/// ```text
///   APE_DESCRIPTOR   cID[4] nVersion:u16 nPadding:u16 nDescriptorBytes:u32
///                    nHeaderBytes:u32 nSeekTableBytes:u32 nHeaderDataBytes:u32
///                    nAPEFrameDataBytes:u32 nAPEFrameDataBytesHigh:u32
///                    nTerminatingDataBytes:u32 cFileMD5[16]
///   APE_HEADER       nCompressionLevel:u16 nFormatFlags:u16 nBlocksPerFrame:u32
///                    nFinalFrameBlocks:u32 nTotalFrames:u32 nBitsPerSample:u16
///                    nChannels:u16 nSampleRate:u32
/// ```
///
/// `nPadding` at octet 6 is not decoration — it exists "because 4-byte alignment requires this
/// (or else nVersion would take 4-bytes)" — and skipping it would shift every later field.
///
/// The header is reached by `nDescriptorBytes`, not by `sizeof(APE_DESCRIPTOR)`: that field is
/// the format's forward-compatibility hinge. A value below the known size is a corrupt file
/// and is clamped up rather than trusted, so the walk can never move backwards.
fn modern(h: &[u8], props: &mut Props) {
    let Some(descriptor_bytes) = le32(h, 8) else { return };
    let header_at = usize::try_from(descriptor_bytes).unwrap_or(DESCRIPTOR_LEN).max(DESCRIPTOR_LEN);
    let Some(a) = h.get(header_at..).filter(|r| r.len() >= HEADER_LEN) else { return };

    let (Some(level), Some(blocks_per_frame), Some(final_frame_blocks), Some(total_frames)) =
        (le16(a, 0), le32(a, 4), le32(a, 8), le32(a, 12))
    else {
        return;
    };
    let (Some(channels), Some(rate)) = (le16(a, 18), le32(a, 20)) else { return };

    if !plausible(level, blocks_per_frame, final_frame_blocks, channels) {
        return;
    }
    props.channels = Some(u32::from(channels));
    finish(props, rate, total_blocks(total_frames, blocks_per_frame, final_frame_blocks));
}

/// Version < 3980: one combined `APE_HEADER_OLD` (`spec/APE.md`, "Version < 3980").
///
/// ```text
///   cID[4] nVersion:u16 nCompressionLevel:u16 nFormatFlags:u16 nChannels:u16
///   nSampleRate:u32 nHeaderBytes:u32 nTerminatingBytes:u32 nTotalFrames:u32
///   nFinalFrameBlocks:u32
/// ```
///
/// Neither `nBlocksPerFrame` nor `nBitsPerSample` is stored; both are derived. Only the first
/// is needed for a duration — [`Props`] has no sample-depth field — but the derivation is the
/// whole reason this path exists, so it is spelled out.
fn legacy(h: &[u8], version: u16, props: &mut Props) {
    if h.len() < OLD_HEADER_LEN {
        return;
    }
    let (Some(level), Some(flags), Some(channels), Some(rate)) =
        (le16(h, 6), le16(h, 8), le16(h, 10), le32(h, 12))
    else {
        return;
    };
    let (Some(total_frames), Some(final_frame_blocks)) = (le32(h, 24), le32(h, 28)) else {
        return;
    };

    // The SDK's derivation, verbatim: 9216 blocks per frame before 3.90 (and for 3.80–3.89 at
    // anything but extra-high), 73728 from 3.90, and four times that from 3.95 — the last
    // assignment being unconditional, so it overrides the first.
    let mut blocks_per_frame =
        if version >= 3900 || (version >= 3800 && level == 4000) { 73_728 } else { 9_216 };
    if version >= 3950 {
        blocks_per_frame = 73_728 * 4;
    }
    // `nBitsPerSample` would be 8 for FLAG_8_BIT, 24 for FLAG_24_BIT, else 16. Read here only
    // to document the flags' meaning; `Props` carries no bit depth.
    let _depth = if flags & FLAG_8_BIT != 0 {
        8
    } else if flags & FLAG_24_BIT != 0 {
        24
    } else {
        16
    };

    if !plausible(level, blocks_per_frame, final_frame_blocks, channels) {
        return;
    }
    // "fail on 0 length APE files (catches non-finalized APE files)" — the SDK refuses these
    // outright on the old path, where the modern path merely reports zero blocks. Matched, so
    // a truncated 1999 file is reported as having no duration rather than a duration of zero.
    if total_frames == 0 {
        return;
    }
    props.channels = Some(u32::from(channels));
    finish(props, rate, total_blocks(total_frames, blocks_per_frame, final_frame_blocks));
}

/// The SDK's sanity gate on a header, shared by both paths.
fn plausible(level: u16, blocks_per_frame: u32, final_frame_blocks: u32, channels: u16) -> bool {
    let ceiling =
        if level >= LEVEL_INSANE { MAX_BLOCKS_PER_FRAME_INSANE } else { MAX_BLOCKS_PER_FRAME };
    blocks_per_frame > 0
        && blocks_per_frame <= ceiling
        && final_frame_blocks <= blocks_per_frame
        && (1..=MAX_CHANNELS).contains(&channels)
}

/// Total sample frames in the file.
///
/// ```text
///   nTotalBlocks = (nTotalFrames == 0) ? 0
///                : (nTotalFrames - 1) * nBlocksPerFrame + nFinalFrameBlocks
/// ```
///
/// The zero guard is the SDK's and is load-bearing: without it `nTotalFrames - 1` underflows
/// to `0xFFFFFFFF` and an unfinalised or hostile file claims about 2.8 million years.
fn total_blocks(total_frames: u32, blocks_per_frame: u32, final_frame_blocks: u32) -> u64 {
    if total_frames == 0 {
        return 0;
    }
    u64::from(total_frames - 1) * u64::from(blocks_per_frame) + u64::from(final_frame_blocks)
}

/// `duration = nTotalBlocks / nSampleRate`, exact — a declared block count, not a bitrate
/// estimate. A zero rate yields no duration rather than a division by zero.
fn finish(props: &mut Props, rate: u32, blocks: u64) {
    if rate == 0 {
        return;
    }
    props.sample_rate = Some(rate);
    // u128: `blocks * 1e9` overflows u64 past ~18 G frames.
    props.duration_ns = u64::try_from(u128::from(blocks) * 1_000_000_000 / u128::from(rate)).ok();
    props.duration_exact = true;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use crate::fixture::{ape_file, ape_file_old};

    #[test]
    fn modern_header_props_and_exact_duration() {
        // 44100 blocks per frame is not a real value; the real ones are big. Two whole frames
        // of 73728 plus a 4644-block final frame is exactly 152100 blocks — 3.45 s at 44.1 kHz.
        let f = ape_file(3990, 44_100, 2, 73_728, 3, 4_644);
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(44_100));
        assert_eq!(p.channels, Some(2));
        let blocks = 2 * 73_728 + 4_644;
        assert_eq!(p.duration_ns, Some(blocks * 1_000_000_000 / 44_100));
        assert!(p.duration_exact, "a declared block count is authoritative");
    }

    #[test]
    fn macf_is_the_same_layout() {
        let mut f = ape_file(3990, 48_000, 1, 73_728, 2, 73_728);
        f[..4].copy_from_slice(MAGIC_LARGE);
        let p = props(&f, 0);
        assert_eq!(p.sample_rate, Some(48_000));
        assert_eq!(p.channels, Some(1));
        assert_eq!(p.duration_ns, Some(2 * 73_728 * 1_000_000_000 / 48_000));
    }

    /// The descriptor states its own length so a later SDK may grow it. A reader that hopped
    /// by a hard-coded 52 would read the header from the wrong offset.
    #[test]
    fn the_header_is_reached_by_the_declared_descriptor_size() {
        let mut f = ape_file(3990, 44_100, 2, 73_728, 2, 73_728);
        // Grow the descriptor by 8 octets and move the header along with it.
        let grown = (DESCRIPTOR_LEN + 8) as u32;
        f[8..12].copy_from_slice(&grown.to_le_bytes());
        let header: Vec<u8> = f[DESCRIPTOR_LEN..DESCRIPTOR_LEN + HEADER_LEN].to_vec();
        let mut g = f[..DESCRIPTOR_LEN].to_vec();
        g.extend_from_slice(&[0u8; 8]); // the eight new descriptor octets
        g.extend_from_slice(&header);
        let p = props(&g, 0);
        assert_eq!(p.sample_rate, Some(44_100));
        assert_eq!(p.duration_ns, Some(2 * 73_728 * 1_000_000_000 / 44_100));

        // A descriptor claiming to be smaller than the known struct is corrupt; clamping up
        // must not make the parse read the descriptor as the header.
        let mut small = f.clone();
        small[8..12].copy_from_slice(&4u32.to_le_bytes());
        let p = props(&small, 0);
        assert_eq!(p.sample_rate, Some(44_100), "clamped to the known size");
    }

    /// The three blocks-per-frame tiers a pre-3.98 file derives rather than stores.
    #[test]
    fn the_old_header_derives_its_blocks_per_frame() {
        // < 3800 at normal compression: 9216.
        let f = ape_file_old(3790, 2000, 0, 44_100, 2, 4, 9_216);
        assert_eq!(props(&f, 0).duration_ns, Some(4 * 9_216 * 1_000_000_000 / 44_100));
        // 3800 at extra-high (4000): 73728.
        let g = ape_file_old(3800, 4000, 0, 44_100, 2, 2, 73_728);
        assert_eq!(props(&g, 0).duration_ns, Some(2 * 73_728 * 1_000_000_000 / 44_100));
        // 3800 at normal: still 9216.
        let h = ape_file_old(3800, 2000, 0, 44_100, 2, 4, 9_216);
        assert_eq!(props(&h, 0).duration_ns, Some(4 * 9_216 * 1_000_000_000 / 44_100));
        // >= 3900: 73728.
        let i = ape_file_old(3900, 2000, 0, 44_100, 2, 2, 73_728);
        assert_eq!(props(&i, 0).duration_ns, Some(2 * 73_728 * 1_000_000_000 / 44_100));
        // >= 3950: 294912, overriding the tier above.
        let j = ape_file_old(3950, 2000, 0, 44_100, 2, 2, 294_912);
        assert_eq!(props(&j, 0).duration_ns, Some(2 * 294_912 * 1_000_000_000 / 44_100));
        assert_eq!(props(&j, 0).channels, Some(2));
        assert_eq!(props(&j, 0).sample_rate, Some(44_100));
    }

    /// The whole point of the SDK's guard: `nTotalFrames == 0` must not underflow.
    #[test]
    fn zero_total_frames_never_underflows() {
        assert_eq!(total_blocks(0, 73_728, 1_000), 0);
        let f = ape_file(3990, 44_100, 2, 73_728, 0, 0);
        let p = props(&f, 0);
        assert_eq!(p.duration_ns, Some(0), "zero blocks, not 2.8 million years");
        assert_eq!(p.sample_rate, Some(44_100));
        // The old path refuses the file outright instead.
        let g = ape_file_old(3950, 2000, 0, 44_100, 2, 0, 294_912);
        assert_eq!(props(&g, 0), Props::default());
    }

    #[test]
    fn implausible_headers_report_nothing() {
        // blocks_per_frame of zero.
        let a = ape_file(3990, 44_100, 2, 0, 4, 0);
        assert_eq!(props(&a, 0), Props::default());
        // blocks_per_frame past the ordinary ceiling at a non-insane level.
        let b = ape_file(3990, 44_100, 2, 2_000_000, 4, 100);
        assert_eq!(props(&b, 0), Props::default());
        // …but legal at insane.
        let c = ape_file(3990, 44_100, 2, 2_000_000, 4, 100);
        let mut c = c;
        c[DESCRIPTOR_LEN..DESCRIPTOR_LEN + 2].copy_from_slice(&LEVEL_INSANE.to_le_bytes());
        assert_eq!(props(&c, 0).sample_rate, Some(44_100));
        // final frame longer than a whole frame.
        let d = ape_file(3990, 44_100, 2, 73_728, 4, 73_729);
        assert_eq!(props(&d, 0), Props::default());
        // zero and out-of-range channel counts.
        let e = ape_file(3990, 44_100, 0, 73_728, 4, 100);
        assert_eq!(props(&e, 0), Props::default());
        let g = ape_file(3990, 44_100, 33, 73_728, 4, 100);
        assert_eq!(props(&g, 0), Props::default());
        // a zero sample rate divides by nothing.
        let i = ape_file(3990, 0, 2, 73_728, 4, 100);
        assert_eq!(props(&i, 0).duration_ns, None);
    }

    #[test]
    fn a_bad_magic_yields_nothing() {
        let mut f = ape_file(3990, 44_100, 2, 73_728, 2, 100);
        f[0] = b'X';
        assert_eq!(props(&f, 0), Props::default());
        assert_eq!(props(b"", 0), Props::default());
        assert_eq!(props(b"MAC ", 0), Props::default());
    }

    #[test]
    fn truncation_never_panics() {
        for f in [ape_file(3990, 44_100, 2, 73_728, 9, 1_000), ape_file_old(3950, 2000, 4, 44_100, 2, 9, 294_912)] {
            for n in 0..f.len() {
                let _ = props(&f[..n], 0);
            }
        }
    }

    #[test]
    fn single_byte_corruption_never_panics() {
        for base in [ape_file(3990, 44_100, 2, 73_728, 9, 1_000), ape_file_old(3800, 4000, 20, 48_000, 1, 5, 73_728)] {
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
