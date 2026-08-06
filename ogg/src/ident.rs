//! Codec identification headers carried on an Ogg bos page: Opus (RFC 7845 §5.1), Vorbis
//! (Vorbis I specification §4.2.2) and Speex (Speex manual §7.3 / Table 7.1, field list per
//! `spec/speex_header.h`; interpretation notes in `spec/SPEEX.md`).
//!
//! An Ogg bos page holds exactly the codec's identification packet (RFC 3533 §6), so
//! `channels` / `sample_rate` / `pre_skip` — everything a probe, a tag scanner, or a
//! duration calculation needs — is readable from the first page of the file without
//! decoding a single audio sample.
//!
//! ## Why the container crate parses these itself
//! `pf-ogg` deliberately depends on **no codec crate** (see the crate docs): a scanner that
//! wants "44100 Hz, 2 channels, 3:41" must not drag in an Opus or Vorbis decoder — with its
//! tables, its state, and in the Opus encoder's case a statically linked C library — to read
//! nineteen fixed bytes. Both headers below are flat, fully specified structs; parsing them
//! here keeps the container plugin codec-agnostic and the scanner dependency-free. The codec
//! crates remain the authority for *decoding*; this is metadata only.
//!
//! Nothing here allocates, and nothing here panics on hostile input: every field is read
//! through a bounds-checked accessor and any short or malformed packet yields `None`.

/// The Opus identification header (RFC 7845 §5.1), as much of it as a container needs.
///
/// The full header also carries an output gain (Q7.8 dB) and a channel mapping family with
/// its optional mapping table; those steer *decoding* and belong to the decoder, not to the
/// container's metadata view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpusHead {
    /// Channel count (§5.1: "MUST be greater than zero").
    pub channels: u8,
    /// Samples to discard from the decoder output at start of playback, at 48 kHz (§5.1).
    /// Also the offset subtracted from the last granule position to get the true duration
    /// (§4) — see [`opus_duration_ns`](crate::duration::opus_duration_ns).
    pub pre_skip: u16,
    /// The **original** input sample rate in Hz (§5.1). Informational only: "This field is
    /// not the sample rate to use for playback of the encoded data" — Opus always decodes at
    /// 48 kHz — but it is what a tag display should show. 0 means "unspecified".
    pub input_sample_rate: u32,
}

/// Fixed size of the Opus ID header up to and including the channel mapping family byte
/// (RFC 7845 §5.1): magic(8) + version(1) + channels(1) + pre-skip(2) + input rate(4) +
/// output gain(2) + mapping family(1).
const OPUS_HEAD_MIN: usize = 19;

/// Parse an Opus identification header packet (RFC 7845 §5.1). `None` if the magic
/// `"OpusHead"` is absent, the packet is short, the version is not backwards-compatible, or
/// the channel count is zero.
///
/// Version handling follows §5.1: "The version number MUST always be '1' ... implementations
/// SHOULD treat streams where the upper four bits of the version number are 0 as backwards
/// compatible" — so the major version (the high nibble) must be 0 and the minor version is
/// ignored.
pub fn parse_opus_head(packet: &[u8]) -> Option<OpusHead> {
    let rest = packet.strip_prefix(b"OpusHead")?;
    if rest.len() < OPUS_HEAD_MIN - 8 {
        return None;
    }
    if rest[0] >> 4 != 0 {
        return None; // incompatible major version
    }
    let channels = rest[1];
    if channels == 0 {
        return None;
    }
    // Multi-byte fields are little-endian (§5.1: "octets ... stored in little-endian order").
    Some(OpusHead {
        channels,
        pre_skip: u16::from_le_bytes([rest[2], rest[3]]),
        input_sample_rate: u32::from_le_bytes([rest[4], rest[5], rest[6], rest[7]]),
    })
}

/// The Vorbis identification header (Vorbis I §4.2.2), as much of it as a container needs.
///
/// The full header also carries the version, three bitrate hints and the two block sizes;
/// those steer decoding and bitrate display, not the container's metadata view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VorbisIdent {
    /// Channel count (§4.2.2: "the bitstream is undecodable" if zero).
    pub channels: u8,
    /// Sample rate in Hz — and the unit of the Vorbis granule position, so it is what turns
    /// a granule into a duration ([`vorbis_duration_ns`](crate::duration::vorbis_duration_ns)).
    pub sample_rate: u32,
}

/// Fixed size of the Vorbis identification packet (Vorbis I §4.2.2): `0x01 "vorbis"`(7) +
/// version(4) + channels(1) + rate(4) + three bitrates(12) + blocksizes(1) + framing bit(1).
const VORBIS_IDENT_LEN: usize = 30;

/// Parse a Vorbis identification header packet: packet type `0x01` and the codec identifier
/// `"vorbis"` (Vorbis I §4.2.1), then the fields of §4.2.2. `None` if the packet is not a
/// Vorbis identification header, is shorter than the header's fixed 30 bytes, declares a
/// non-zero `vorbis_version`, or violates §4.2.2's validity rules ("the bitstream is
/// undecodable" if `audio_channels` or `audio_sample_rate` is zero).
///
/// `vorbis_version` "is to read 0 in order to be compatible with this document" (§4.2.2), so
/// a non-zero version is a stream this parser must not claim to understand.
pub fn parse_vorbis_ident(packet: &[u8]) -> Option<VorbisIdent> {
    let rest = packet.strip_prefix(b"\x01vorbis")?;
    if packet.len() < VORBIS_IDENT_LEN {
        return None;
    }
    // Vorbis packs its header integers LSB-first (§1.2.2 "byte order"), like Ogg itself.
    if u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]) != 0 {
        return None;
    }
    let channels = rest[4];
    let sample_rate = u32::from_le_bytes([rest[5], rest[6], rest[7], rest[8]]);
    if channels == 0 || sample_rate == 0 {
        return None;
    }
    Some(VorbisIdent { channels, sample_rate })
}

/// The Speex identification header (Speex manual §7.3, Table 7.1), as much of it as a
/// container needs.
///
/// The full 80-byte header also carries the encoder version string, the mode
/// (narrowband/wideband/ultra-wideband) and its bitstream version, the bitrate, the frame
/// size, the VBR flag, `frames_per_packet` and `extra_headers`; those steer *decoding* and
/// belong to the decoder, not to the container's metadata view. See `spec/SPEEX.md` for the
/// complete field table and for why `extra_headers` cannot move the comment packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeexHead {
    /// Sample rate in Hz — and the unit of the Speex granule position, so it is what turns a
    /// granule into a duration ([`vorbis_duration_ns`](crate::duration::vorbis_duration_ns),
    /// shared with the Vorbis and FLAC mappings).
    pub sample_rate: u32,
    /// Channel count (`nb_channels`).
    pub channels: u8,
}

/// The magic that identifies a Speex bitstream: `"Speex"` and **three** trailing spaces
/// (Speex manual §7.3 — "must contain the 'Speex   ' (with 3 trailing spaces)").
pub const SPEEX_MAGIC: &[u8; 8] = b"Speex   ";

/// Fixed size of the Speex ID header (`spec/speex_header.h`): `speex_string`(8) +
/// `speex_version`(20) + thirteen 32-bit fields (52) = 80 octets.
const SPEEX_HEAD_LEN: usize = 80;

/// Byte offset of `rate` within the packet: magic(8) + version string(20) +
/// `speex_version_id`(4) + `header_size`(4).
const SPEEX_RATE_AT: usize = 36;

/// Byte offset of `nb_channels`: [`SPEEX_RATE_AT`] + `rate`(4) + `mode`(4) +
/// `mode_bitstream_version`(4).
const SPEEX_CHANNELS_AT: usize = 48;

/// Parse a Speex identification header packet (Speex manual §7.3, Table 7.1). `None` if the
/// magic `"Speex   "` is absent, the packet is shorter than the header's fixed 80 octets, or
/// the stream is undecodable — a zero sample rate or zero channel count.
///
/// A channel count is rejected above [`u8::MAX`] as well as at zero: `nb_channels` is a
/// *signed* 32-bit field (`spx_int32_t`), so a corrupt header can hold a negative or absurd
/// value, and a container must not launder that into a plausible-looking property.
pub fn parse_speex_head(packet: &[u8]) -> Option<SpeexHead> {
    if !packet.starts_with(SPEEX_MAGIC) || packet.len() < SPEEX_HEAD_LEN {
        return None;
    }
    // "All integer fields in the headers are stored as little-endian" (§7.3).
    let le32 = |at: usize| -> i32 {
        let b = &packet[at..at + 4];
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };
    let sample_rate = u32::try_from(le32(SPEEX_RATE_AT)).ok().filter(|&r| r > 0)?;
    let channels = u8::try_from(le32(SPEEX_CHANNELS_AT)).ok().filter(|&c| c > 0)?;
    Some(SpeexHead { sample_rate, channels })
}

#[cfg(test)]
// Tests build header fixtures; the parsers under test allocate nothing (spec: allocation
// discipline — tests are the sanctioned exception).
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    /// An Opus ID header (RFC 7845 §5.1) with mapping family 0.
    fn opus_head(version: u8, channels: u8, pre_skip: u16, rate: u32) -> Vec<u8> {
        let mut p = b"OpusHead".to_vec();
        p.push(version);
        p.push(channels);
        p.extend_from_slice(&pre_skip.to_le_bytes());
        p.extend_from_slice(&rate.to_le_bytes());
        p.extend_from_slice(&0i16.to_le_bytes()); // output gain
        p.push(0); // channel mapping family
        p
    }

    /// A Vorbis identification packet (Vorbis I §4.2.2).
    fn vorbis_ident(version: u32, channels: u8, rate: u32) -> Vec<u8> {
        let mut p = b"\x01vorbis".to_vec();
        p.extend_from_slice(&version.to_le_bytes());
        p.push(channels);
        p.extend_from_slice(&rate.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes()); // bitrate maximum
        p.extend_from_slice(&192_000u32.to_le_bytes()); // bitrate nominal
        p.extend_from_slice(&0u32.to_le_bytes()); // bitrate minimum
        p.push(0xB8); // blocksize_0 = 2^8, blocksize_1 = 2^11
        p.push(0x01); // framing flag
        p
    }

    /// A Speex identification packet (Speex manual Table 7.1): the magic, a 20-byte version
    /// string, then thirteen little-endian 32-bit fields.
    fn speex_head(rate: i32, channels: i32, extra_headers: i32) -> Vec<u8> {
        let mut p = SPEEX_MAGIC.to_vec();
        let mut version = [0u8; 20];
        version[..5].copy_from_slice(b"1.2rc");
        p.extend_from_slice(&version);
        for f in [
            1,             // speex_version_id
            80,            // header_size
            rate,          // rate
            1,             // mode: wideband
            4,             // mode_bitstream_version
            channels,      // nb_channels
            27_800,        // bitrate
            320,           // frame_size
            0,             // vbr
            1,             // frames_per_packet
            extra_headers, // extra_headers
            0,             // reserved1
            0,             // reserved2
        ] {
            p.extend_from_slice(&f.to_le_bytes());
        }
        p
    }

    #[test]
    fn parses_speex_head() {
        let head = parse_speex_head(&speex_head(16_000, 1, 0)).expect("valid Speex header");
        assert_eq!(head, SpeexHead { sample_rate: 16_000, channels: 1 });
        assert_eq!(speex_head(16_000, 1, 0).len(), SPEEX_HEAD_LEN, "Table 7.1 is 80 octets");
        // `extra_headers` steers only where the *audio* starts; it must not change the header.
        assert_eq!(
            parse_speex_head(&speex_head(8_000, 2, 3)),
            Some(SpeexHead { sample_rate: 8_000, channels: 2 })
        );
        // A real ffmpeg/libspeex bos payload: 16 kHz mono wideband (see `spec/SPEEX.md`).
        let mut real = SPEEX_MAGIC.to_vec();
        real.extend_from_slice(b"speex-1.2rc1\0\0\0\0\0\0\0\0");
        real.extend_from_slice(&1i32.to_le_bytes());
        real.extend_from_slice(&80i32.to_le_bytes());
        real.extend_from_slice(&16_000i32.to_le_bytes());
        real.extend_from_slice(&[0u8; 44]);
        // Patch nb_channels in place, at the offset the field table gives.
        real[SPEEX_CHANNELS_AT..SPEEX_CHANNELS_AT + 4].copy_from_slice(&1i32.to_le_bytes());
        assert_eq!(
            parse_speex_head(&real),
            Some(SpeexHead { sample_rate: 16_000, channels: 1 })
        );
    }

    #[test]
    fn rejects_bad_speex_head() {
        assert!(parse_speex_head(&speex_head(0, 1, 0)).is_none()); // zero rate
        assert!(parse_speex_head(&speex_head(16_000, 0, 0)).is_none()); // zero channels
        // `nb_channels` is signed: a negative or absurd count must not become a property.
        assert!(parse_speex_head(&speex_head(16_000, -2, 0)).is_none());
        assert!(parse_speex_head(&speex_head(16_000, 4096, 0)).is_none());
        assert!(parse_speex_head(&speex_head(-8_000, 1, 0)).is_none());
        // Two trailing spaces, not three — not a Speex stream.
        let mut wrong = speex_head(16_000, 1, 0);
        wrong[7] = b'!';
        assert!(parse_speex_head(&wrong).is_none());
        assert!(parse_speex_head(b"OpusHead\x01\x02").is_none());
        let full = speex_head(16_000, 2, 0);
        for n in 0..full.len() {
            assert!(parse_speex_head(&full[..n]).is_none(), "truncated to {n} must not parse");
        }
    }

    #[test]
    fn parses_opus_head() {
        let head = parse_opus_head(&opus_head(1, 2, 312, 44_100)).expect("valid OpusHead");
        assert_eq!(head, OpusHead { channels: 2, pre_skip: 312, input_sample_rate: 44_100 });
        // Version 0 and a minor-version bump both stay backwards compatible (§5.1).
        assert!(parse_opus_head(&opus_head(0, 1, 0, 0)).is_some());
        assert!(parse_opus_head(&opus_head(0x0F, 1, 0, 0)).is_some());
    }

    #[test]
    fn rejects_bad_opus_head() {
        assert!(parse_opus_head(&opus_head(0x10, 2, 0, 48_000)).is_none()); // major version 1
        assert!(parse_opus_head(&opus_head(1, 0, 0, 48_000)).is_none()); // zero channels
        assert!(parse_opus_head(b"OpusTags\x01\x02").is_none()); // wrong magic
        assert!(parse_opus_head(b"").is_none());
        // Every truncation of a valid header is rejected, and none of them panics.
        let full = opus_head(1, 2, 312, 48_000);
        for n in 0..full.len() {
            assert!(parse_opus_head(&full[..n]).is_none(), "truncated to {n} must not parse");
        }
    }

    #[test]
    fn parses_vorbis_ident() {
        let id = parse_vorbis_ident(&vorbis_ident(0, 2, 44_100)).expect("valid ident");
        assert_eq!(id, VorbisIdent { channels: 2, sample_rate: 44_100 });
        // The real libVorbis bos payload from `page.rs`'s known-answer test: 2ch, 44100 Hz.
        const PAYLOAD: [u8; 30] = [
            0x01, 0x76, 0x6f, 0x72, 0x62, 0x69, 0x73, 0x00, 0x00, 0x00, 0x00, 0x02, 0x44, 0xac,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xb5, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xb8, 0x01,
        ];
        assert_eq!(
            parse_vorbis_ident(&PAYLOAD),
            Some(VorbisIdent { channels: 2, sample_rate: 44_100 })
        );
    }

    #[test]
    fn rejects_bad_vorbis_ident() {
        assert!(parse_vorbis_ident(&vorbis_ident(1, 2, 44_100)).is_none()); // version != 0
        assert!(parse_vorbis_ident(&vorbis_ident(0, 0, 44_100)).is_none()); // zero channels
        assert!(parse_vorbis_ident(&vorbis_ident(0, 2, 0)).is_none()); // zero rate
        assert!(parse_vorbis_ident(b"\x03vorbis........................").is_none()); // comment hdr
        let full = vorbis_ident(0, 2, 48_000);
        for n in 0..full.len() {
            assert!(parse_vorbis_ident(&full[..n]).is_none(), "truncated to {n} must not parse");
        }
    }
}
