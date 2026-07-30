//! Opus packet TOC byte parser.
//!
//! Implements the Table-of-Contents header described in RFC 6716 §3.1.
//! The first byte of every well-formed Opus packet contains:
//!
//! ```text
//!                              0
//!                              0 1 2 3 4 5 6 7
//!                             +-+-+-+-+-+-+-+-+
//!                             | config  |s| c |
//!                             +-+-+-+-+-+-+-+-+
//! ```
//!
//! * `config` — five bits (bit 0 is the MSB per RFC 6716 numbering),
//!   selecting one of 32 (mode, bandwidth, frame-size) tuples per
//!   Table 2.
//! * `s` — one bit: 0 = mono, 1 = stereo per Table 3 (informal —
//!   described in prose immediately after Table 2 in the RFC).
//! * `c` — two bits: code 0..3, the frame-packing code per Table 4
//!   (also described in the prose at the end of §3.1).
//!
//! Only the TOC byte interpretation is implemented here. Frame
//! packing per §3.2 and the SILK / CELT inner decoders are out of
//! scope for round 1.

use core::fmt;

use crate::Error;

/// Operating mode selected by the `config` field of the TOC byte.
///
/// Per RFC 6716 §3.1 Table 2 the 32 configurations cluster into
/// three operating modes: a SILK-only LP mode, a Hybrid SILK+CELT
/// mode, and a CELT-only MDCT mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// LP (SILK) only — used at low bitrates up to WB.
    SilkOnly,
    /// Hybrid SILK + CELT — used for SWB / FB speech at medium
    /// bitrates.
    Hybrid,
    /// CELT (MDCT) only — used for very low delay and music
    /// transmission.
    CeltOnly,
}

/// Audio bandwidth selected by the `config` field of the TOC byte.
///
/// The five bandwidths per RFC 6716 §2:
///
/// * NB — narrowband, 4 kHz effective, 8 kHz sample rate equivalent.
/// * MB — medium band, 6 kHz effective.
/// * WB — wideband, 8 kHz effective.
/// * SWB — super-wideband, 12 kHz effective.
/// * FB — fullband, 20 kHz effective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bandwidth {
    Nb,
    Mb,
    Wb,
    Swb,
    Fb,
}

/// Channel mapping signalled by the `s` bit of the TOC byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelMapping {
    /// `s = 0` — one channel.
    Mono,
    /// `s = 1` — two channels.
    Stereo,
}

/// Frame-packing code signalled by the `c` field of the TOC byte
/// (RFC 6716 §3.1, immediately after Table 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameCountCode {
    /// Code 0 — exactly one frame in the packet.
    One,
    /// Code 1 — exactly two frames, both compressed to the same size.
    TwoEqual,
    /// Code 2 — exactly two frames with independent compressed sizes.
    TwoUnequal,
    /// Code 3 — arbitrary frame count, encoded in a following byte.
    Arbitrary,
}

/// Decoded interpretation of a single Opus packet TOC byte.
///
/// This does not consume any bytes beyond the TOC byte itself — frame
/// packing (the second byte for code 3, the length sequence for
/// code 2, etc.) is the §3.2 layer and lives elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpusTocByte {
    /// Raw `config` value (0..31).
    pub config: u8,
    /// Operating mode selected by `config`.
    pub mode: Mode,
    /// Audio bandwidth selected by `config`.
    pub bandwidth: Bandwidth,
    /// Per-frame duration in tenths of a millisecond.
    ///
    /// Encoded in tenths because RFC 6716 allows 2.5 ms (= 25) which
    /// is not representable in whole milliseconds. The legal values
    /// per Table 2 are 25, 50, 100, 200, 400, 600.
    pub frame_size_tenths_ms: u16,
    /// Channel mapping signalled by `s`.
    pub channels: ChannelMapping,
    /// Frame-packing code signalled by `c`.
    pub frame_count_code: FrameCountCode,
}

impl OpusTocByte {
    /// Parse the TOC byte from the front of an Opus packet.
    ///
    /// Returns [`Error::EmptyPacket`] if `packet` is empty (RFC 6716
    /// §3.1 requirement R1). Any non-empty packet has a syntactically
    /// valid TOC byte by construction (every five-bit `config` value
    /// is assigned, every `s` and `c` value is assigned), so the only
    /// failure mode is the empty-packet case.
    pub fn parse(packet: &[u8]) -> Result<Self, Error> {
        let first = *packet.first().ok_or(Error::EmptyPacket)?;
        Ok(Self::from_byte(first))
    }

    /// Decode an isolated TOC byte. Total function — every `u8` is a
    /// valid TOC byte.
    pub fn from_byte(byte: u8) -> Self {
        // RFC 6716 §3.1 numbers bit 0 as the most significant bit.
        // In u8 terms (bit 7 = MSB): config is the top 5 bits,
        // s is bit 2 (counted from LSB), c is the low 2 bits.
        let config = byte >> 3;
        let s = (byte >> 2) & 0x01;
        let c = byte & 0x03;

        let (mode, bandwidth, frame_size_tenths_ms) = decode_config(config);
        let channels = if s == 0 {
            ChannelMapping::Mono
        } else {
            ChannelMapping::Stereo
        };
        let frame_count_code = match c {
            0 => FrameCountCode::One,
            1 => FrameCountCode::TwoEqual,
            2 => FrameCountCode::TwoUnequal,
            3 => FrameCountCode::Arbitrary,
            _ => unreachable!("c is masked to 2 bits"),
        };

        Self {
            config,
            mode,
            bandwidth,
            frame_size_tenths_ms,
            channels,
            frame_count_code,
        }
    }

    /// Compose the TOC byte for a `(mode, bandwidth, frame size)`
    /// triple plus the channel flag and frame-count code — the inverse
    /// of [`Self::from_byte`] (RFC 6716 §3.1).
    ///
    /// Returns [`Error::MalformedPacket`] when the triple does not
    /// correspond to any Table-2 `config` row (e.g. SILK-only SWB, or
    /// a CELT-only 40 ms frame).
    pub fn compose_byte(
        mode: Mode,
        bandwidth: Bandwidth,
        frame_size_tenths_ms: u16,
        stereo: bool,
        frame_count_code: FrameCountCode,
    ) -> Result<u8, Error> {
        // Table 2 is small; find the config row by exhaustive match
        // against the decode mapping (correct by construction).
        let config = (0u8..32)
            .find(|&c| decode_config(c) == (mode, bandwidth, frame_size_tenths_ms))
            .ok_or(Error::MalformedPacket)?;
        let s = u8::from(stereo);
        let c = match frame_count_code {
            FrameCountCode::One => 0u8,
            FrameCountCode::TwoEqual => 1,
            FrameCountCode::TwoUnequal => 2,
            FrameCountCode::Arbitrary => 3,
        };
        Ok((config << 3) | (s << 2) | c)
    }

    /// Minimum and maximum frame count the TOC byte implies *without*
    /// consulting subsequent bytes. Codes 0/1/2 have a known frame
    /// count; code 3 needs the §3.2.5 frame-count byte to resolve and
    /// returns the legal `(1, 48)` range here.
    pub fn frame_count_range(self) -> (u8, u8) {
        match self.frame_count_code {
            FrameCountCode::One => (1, 1),
            FrameCountCode::TwoEqual | FrameCountCode::TwoUnequal => (2, 2),
            // RFC 6716 §3.2.5: the M field is 1..48 inclusive (R5).
            FrameCountCode::Arbitrary => (1, 48),
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Mode::SilkOnly => "SILK-only",
            Mode::Hybrid => "Hybrid",
            Mode::CeltOnly => "CELT-only",
        })
    }
}

impl fmt::Display for Bandwidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Bandwidth::Nb => "NB",
            Bandwidth::Mb => "MB",
            Bandwidth::Wb => "WB",
            Bandwidth::Swb => "SWB",
            Bandwidth::Fb => "FB",
        })
    }
}

/// Map a five-bit `config` value to the `(mode, bandwidth,
/// frame-size)` triple per RFC 6716 §3.1 Table 2.
///
/// The configuration numbers in each range correspond to the choices
/// of frame size in the same order — e.g. for SILK-only NB
/// (configs 0..=3) the frame sizes 10, 20, 40, 60 ms correspond to
/// configs 0, 1, 2, 3 respectively.
fn decode_config(config: u8) -> (Mode, Bandwidth, u16) {
    debug_assert!(config < 32);
    match config {
        // SILK-only — 10, 20, 40, 60 ms.
        0 => (Mode::SilkOnly, Bandwidth::Nb, 100),
        1 => (Mode::SilkOnly, Bandwidth::Nb, 200),
        2 => (Mode::SilkOnly, Bandwidth::Nb, 400),
        3 => (Mode::SilkOnly, Bandwidth::Nb, 600),
        4 => (Mode::SilkOnly, Bandwidth::Mb, 100),
        5 => (Mode::SilkOnly, Bandwidth::Mb, 200),
        6 => (Mode::SilkOnly, Bandwidth::Mb, 400),
        7 => (Mode::SilkOnly, Bandwidth::Mb, 600),
        8 => (Mode::SilkOnly, Bandwidth::Wb, 100),
        9 => (Mode::SilkOnly, Bandwidth::Wb, 200),
        10 => (Mode::SilkOnly, Bandwidth::Wb, 400),
        11 => (Mode::SilkOnly, Bandwidth::Wb, 600),
        // Hybrid — 10, 20 ms.
        12 => (Mode::Hybrid, Bandwidth::Swb, 100),
        13 => (Mode::Hybrid, Bandwidth::Swb, 200),
        14 => (Mode::Hybrid, Bandwidth::Fb, 100),
        15 => (Mode::Hybrid, Bandwidth::Fb, 200),
        // CELT-only — 2.5, 5, 10, 20 ms.
        16 => (Mode::CeltOnly, Bandwidth::Nb, 25),
        17 => (Mode::CeltOnly, Bandwidth::Nb, 50),
        18 => (Mode::CeltOnly, Bandwidth::Nb, 100),
        19 => (Mode::CeltOnly, Bandwidth::Nb, 200),
        20 => (Mode::CeltOnly, Bandwidth::Wb, 25),
        21 => (Mode::CeltOnly, Bandwidth::Wb, 50),
        22 => (Mode::CeltOnly, Bandwidth::Wb, 100),
        23 => (Mode::CeltOnly, Bandwidth::Wb, 200),
        24 => (Mode::CeltOnly, Bandwidth::Swb, 25),
        25 => (Mode::CeltOnly, Bandwidth::Swb, 50),
        26 => (Mode::CeltOnly, Bandwidth::Swb, 100),
        27 => (Mode::CeltOnly, Bandwidth::Swb, 200),
        28 => (Mode::CeltOnly, Bandwidth::Fb, 25),
        29 => (Mode::CeltOnly, Bandwidth::Fb, 50),
        30 => (Mode::CeltOnly, Bandwidth::Fb, 100),
        31 => (Mode::CeltOnly, Bandwidth::Fb, 200),
        _ => unreachable!("config is masked to 5 bits"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every five-bit `config` produces the correct (mode, bandwidth,
    /// frame-size) triple per RFC 6716 §3.1 Table 2.
    ///
    /// We assemble the TOC byte as `config<<3 | 0<<2 | 0` (mono,
    /// code 0) so the parser sees exactly the `config` we expect.
    #[test]
    fn table2_all_32_configs() {
        // Independently encoded expected triples mirroring Table 2.
        let expected: [(Mode, Bandwidth, u16); 32] = [
            (Mode::SilkOnly, Bandwidth::Nb, 100),
            (Mode::SilkOnly, Bandwidth::Nb, 200),
            (Mode::SilkOnly, Bandwidth::Nb, 400),
            (Mode::SilkOnly, Bandwidth::Nb, 600),
            (Mode::SilkOnly, Bandwidth::Mb, 100),
            (Mode::SilkOnly, Bandwidth::Mb, 200),
            (Mode::SilkOnly, Bandwidth::Mb, 400),
            (Mode::SilkOnly, Bandwidth::Mb, 600),
            (Mode::SilkOnly, Bandwidth::Wb, 100),
            (Mode::SilkOnly, Bandwidth::Wb, 200),
            (Mode::SilkOnly, Bandwidth::Wb, 400),
            (Mode::SilkOnly, Bandwidth::Wb, 600),
            (Mode::Hybrid, Bandwidth::Swb, 100),
            (Mode::Hybrid, Bandwidth::Swb, 200),
            (Mode::Hybrid, Bandwidth::Fb, 100),
            (Mode::Hybrid, Bandwidth::Fb, 200),
            (Mode::CeltOnly, Bandwidth::Nb, 25),
            (Mode::CeltOnly, Bandwidth::Nb, 50),
            (Mode::CeltOnly, Bandwidth::Nb, 100),
            (Mode::CeltOnly, Bandwidth::Nb, 200),
            (Mode::CeltOnly, Bandwidth::Wb, 25),
            (Mode::CeltOnly, Bandwidth::Wb, 50),
            (Mode::CeltOnly, Bandwidth::Wb, 100),
            (Mode::CeltOnly, Bandwidth::Wb, 200),
            (Mode::CeltOnly, Bandwidth::Swb, 25),
            (Mode::CeltOnly, Bandwidth::Swb, 50),
            (Mode::CeltOnly, Bandwidth::Swb, 100),
            (Mode::CeltOnly, Bandwidth::Swb, 200),
            (Mode::CeltOnly, Bandwidth::Fb, 25),
            (Mode::CeltOnly, Bandwidth::Fb, 50),
            (Mode::CeltOnly, Bandwidth::Fb, 100),
            (Mode::CeltOnly, Bandwidth::Fb, 200),
        ];
        for (config, &(mode, bw, dur)) in expected.iter().enumerate() {
            let toc = OpusTocByte::from_byte((config as u8) << 3);
            assert_eq!(toc.config, config as u8, "config field");
            assert_eq!(toc.mode, mode, "config {config}: mode");
            assert_eq!(toc.bandwidth, bw, "config {config}: bandwidth");
            assert_eq!(
                toc.frame_size_tenths_ms, dur,
                "config {config}: frame-size (tenths of ms)"
            );
        }
    }

    /// The `s` bit (bit position 5 from the MSB / bit 2 from the
    /// LSB) toggles mono vs. stereo independently of `config` and
    /// `c`. We sweep `config` and `c` and verify each `s` polarity.
    #[test]
    fn stereo_bit_independent_of_config_and_code() {
        for config in 0u8..32 {
            for code in 0u8..4 {
                let mono = OpusTocByte::from_byte((config << 3) | code);
                let stereo = OpusTocByte::from_byte((config << 3) | (1 << 2) | code);
                assert_eq!(mono.channels, ChannelMapping::Mono);
                assert_eq!(stereo.channels, ChannelMapping::Stereo);
                // Toggling `s` must not bleed into the other fields.
                assert_eq!(mono.config, stereo.config);
                assert_eq!(mono.mode, stereo.mode);
                assert_eq!(mono.bandwidth, stereo.bandwidth);
                assert_eq!(mono.frame_size_tenths_ms, stereo.frame_size_tenths_ms);
                assert_eq!(mono.frame_count_code, stereo.frame_count_code);
            }
        }
    }

    /// The `c` two-bit field selects the frame-packing code per the
    /// four cases enumerated immediately after Table 2.
    #[test]
    fn frame_count_codes() {
        let cases = [
            (0u8, FrameCountCode::One, (1u8, 1u8)),
            (1, FrameCountCode::TwoEqual, (2, 2)),
            (2, FrameCountCode::TwoUnequal, (2, 2)),
            (3, FrameCountCode::Arbitrary, (1, 48)),
        ];
        for (c, expected_code, range) in cases {
            let toc = OpusTocByte::from_byte(c);
            assert_eq!(toc.frame_count_code, expected_code, "c={c}");
            assert_eq!(toc.frame_count_range(), range, "c={c} frame-count range");
        }
    }

    /// Empty packet rejection per RFC 6716 §3.1 R1.
    #[test]
    fn parse_empty_rejects() {
        assert_eq!(OpusTocByte::parse(&[]), Err(Error::EmptyPacket));
    }

    /// A spot-check parse against a hand-assembled packet:
    /// config=13 (Hybrid SWB 20 ms), s=1 (stereo), c=2 (two
    /// unequal frames). Bit layout: `01101 1 10` = 0x6E.
    #[test]
    fn parse_known_byte() {
        let toc = OpusTocByte::parse(&[0x6E, 0x00, 0x00]).unwrap();
        assert_eq!(toc.config, 13);
        assert_eq!(toc.mode, Mode::Hybrid);
        assert_eq!(toc.bandwidth, Bandwidth::Swb);
        assert_eq!(toc.frame_size_tenths_ms, 200);
        assert_eq!(toc.channels, ChannelMapping::Stereo);
        assert_eq!(toc.frame_count_code, FrameCountCode::TwoUnequal);
        // And a second spot-check at the opposite corner: config=31
        // (CELT FB 20 ms), mono, code 0.
        let toc2 = OpusTocByte::from_byte(0xF8);
        assert_eq!(toc2.config, 31);
        assert_eq!(toc2.mode, Mode::CeltOnly);
        assert_eq!(toc2.bandwidth, Bandwidth::Fb);
        assert_eq!(toc2.frame_size_tenths_ms, 200);
        assert_eq!(toc2.channels, ChannelMapping::Mono);
        assert_eq!(toc2.frame_count_code, FrameCountCode::One);
    }
}
