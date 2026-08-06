//! `OpusHead` identification-header parsing — RFC 7845 §5.1 + §5.1.1.
//!
//! Every Ogg-Opus logical stream begins with an *identification header*
//! (the "OpusHead" packet, RFC 7845 §5.1, Figure 2). It is the Opus
//! decoder's *configuration*: how many output channels to produce, how
//! many samples to discard at start-up (pre-skip), the original input
//! sample rate (metadata only), an output gain, and — crucially for
//! multichannel content — the *channel mapping* that ties one or more
//! coded Opus streams to the stream's output channels (RFC 7845 §5.1.1,
//! Figure 3).
//!
//! ## Why this lives in the codec crate
//!
//! `OpusHead` is the codec's own configuration record, not container
//! framing. The Ogg layer (`oxideav-ogg`) merely delivers the raw header
//! *packet bytes*; interpreting those bytes — validating the magic,
//! version, and the §5.1.1 stream/coupled counts, and deciding how the
//! per-stream decoder outputs combine into the final channels — is Opus
//! decoding. The same header is also what a non-Ogg transport (e.g. a
//! Matroska `CodecPrivate`) carries verbatim, so the parser belongs with
//! the codec, like the §3 framing rules already do.
//!
//! ## What this module produces
//!
//! [`OpusHead::parse`] consumes a header packet and returns a fully
//! validated [`OpusHead`] carrying the §5.1 scalar fields plus the
//! §5.1.1 [`ChannelMappingTable`]. For mapping family 0 the table is
//! synthesized from the defaults the RFC pins (N = 1, M = C − 1, the
//! identity channel map) since family 0 omits the on-wire table.
//!
//! ## Provenance
//!
//! RFC 7845 §5.1 (Figure 2, pp. 12–15) for the scalar fields and §5.1.1
//! (Figure 3, pp. 16–18) for the channel-mapping table, including the
//! family-0 / family-1 / family-255 allowed-channel rules and every MUST
//! validation (`Version`, `Output Channel Count`, `Stream Count`,
//! `Coupled Stream Count`, and the `M + N ≤ 255` decoded-channel bound).

/// The fixed 8-octet magic signature `"OpusHead"` (RFC 7845 §5.1, item
/// 1). A valid identification header begins with exactly these bytes.
pub const OPUS_HEAD_MAGIC: &[u8; 8] = b"OpusHead";

/// The minimum byte length of an identification header packet: the
/// 8-octet magic, plus version / channel-count (2), pre-skip (2), input
/// sample rate (4), output gain (2), and the mapping-family octet (1).
/// A family-0 header is exactly this long; family ≥ 1 headers append the
/// §5.1.1 channel-mapping table.
pub const OPUS_HEAD_MIN_LEN: usize = 19;

/// The maximum recognized major version. RFC 7845 §5.1 item 2: an
/// implementation "SHOULD accept any stream with a version number of
/// '15' or less, and SHOULD assume any stream with a version number
/// '16' or greater is incompatible." The major version is the upper
/// nibble, so versions `0x00..=0x0F` are accepted.
pub const OPUS_HEAD_MAX_VERSION: u8 = 15;

/// Errors that can arise parsing an `OpusHead` identification header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusHeadError {
    /// The packet is shorter than the smallest valid header.
    TooShort {
        /// Length supplied.
        got: usize,
        /// Minimum required ([`OPUS_HEAD_MIN_LEN`], or more once the
        /// mapping family demands a channel-mapping table).
        need: usize,
    },
    /// The leading 8 octets are not the [`OPUS_HEAD_MAGIC`] signature.
    BadMagic,
    /// The major version (upper nibble of the version octet) exceeds
    /// [`OPUS_HEAD_MAX_VERSION`]; the stream is from an incompatible
    /// future encapsulation revision (RFC 7845 §5.1 item 2).
    IncompatibleVersion {
        /// The raw version octet.
        version: u8,
    },
    /// `Output Channel Count` is zero. RFC 7845 §5.1 item 3: "This value
    /// MUST NOT be zero."
    ZeroChannels,
    /// The output channel count is out of range for the signalled
    /// mapping family (family 0 allows 1..=2, family 1 allows 1..=8).
    ChannelCountForFamily {
        /// The mapping family octet.
        family: u8,
        /// The output channel count `C`.
        channels: u8,
    },
    /// `Stream Count` N is zero. RFC 7845 §5.1.1 item 1: "This value
    /// MUST NOT be zero."
    ZeroStreams,
    /// `Coupled Stream Count` M is larger than `Stream Count` N. RFC
    /// 7845 §5.1.1 item 2: "This MUST be no larger than the total number
    /// of streams, N."
    CoupledExceedsStreams {
        /// Stream count N.
        streams: u8,
        /// Coupled count M.
        coupled: u8,
    },
    /// `M + N` (the total decoded channel count) exceeds 255. RFC 7845
    /// §5.1.1 item 2: "The total number of decoded channels, (M + N),
    /// MUST be no larger than 255."
    TooManyDecodedChannels {
        /// Stream count N.
        streams: u8,
        /// Coupled count M.
        coupled: u8,
    },
    /// A per-output-channel mapping index is neither `< (M + N)` nor the
    /// reserved silence value 255. RFC 7845 §5.1.1 item 3: each index
    /// "MUST either be smaller than (M + N) or be the special value
    /// 255."
    MappingIndexOutOfRange {
        /// The offending output-channel position.
        output_channel: u8,
        /// The index value found there.
        index: u8,
        /// The decoded-channel bound `M + N`.
        decoded_channels: u8,
    },
}

impl core::fmt::Display for OpusHeadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            OpusHeadError::TooShort { got, need } => {
                write!(f, "OpusHead too short: {got} bytes, need {need}")
            }
            OpusHeadError::BadMagic => write!(f, "OpusHead missing 'OpusHead' magic signature"),
            OpusHeadError::IncompatibleVersion { version } => {
                write!(
                    f,
                    "OpusHead major version {} > {OPUS_HEAD_MAX_VERSION}",
                    version >> 4
                )
            }
            OpusHeadError::ZeroChannels => write!(f, "OpusHead output channel count is zero"),
            OpusHeadError::ChannelCountForFamily { family, channels } => write!(
                f,
                "OpusHead channel count {channels} invalid for mapping family {family}"
            ),
            OpusHeadError::ZeroStreams => write!(f, "OpusHead stream count N is zero"),
            OpusHeadError::CoupledExceedsStreams { streams, coupled } => {
                write!(
                    f,
                    "OpusHead coupled count {coupled} exceeds stream count {streams}"
                )
            }
            OpusHeadError::TooManyDecodedChannels { streams, coupled } => write!(
                f,
                "OpusHead decoded channels M+N = {}+{} exceeds 255",
                coupled, streams
            ),
            OpusHeadError::MappingIndexOutOfRange {
                output_channel,
                index,
                decoded_channels,
            } => write!(
                f,
                "OpusHead mapping index {index} for output channel {output_channel} \
                 is neither < {decoded_channels} nor 255"
            ),
        }
    }
}

impl std::error::Error for OpusHeadError {}

/// The §5.1.1 channel-mapping table: how the `N` coded streams (the
/// first `M` of which are stereo) combine into the stream's `C` output
/// channels.
///
/// For mapping family 0 this is synthesized from the RFC-pinned defaults
/// (the on-wire header omits the table entirely); for families ≥ 1 it is
/// read from the header bytes following the mapping-family octet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelMappingTable {
    /// `N` — total number of coded Opus streams in each Ogg packet
    /// (§5.1.1 item 1). Always ≥ 1.
    pub stream_count: u8,
    /// `M` — number of those streams decoded as stereo (§5.1.1 item 2).
    /// The first `M` streams are stereo; the remaining `N − M` are mono.
    pub coupled_count: u8,
    /// One index per output channel (§5.1.1 item 3). `mapping[c]` selects
    /// the decoded channel feeding output channel `c`:
    ///
    /// * `index < 2*M` → stereo stream `index/2`, left channel if even
    ///   else right.
    /// * `2*M ≤ index < 255` → mono stream `index − M`.
    /// * `index == 255` → pure silence.
    pub mapping: Vec<u8>,
}

impl ChannelMappingTable {
    /// Total decoded channel count `M + N` (RFC 7845 §5.1.1 item 2).
    /// This is the number of distinct decoder channels the streams
    /// produce before the [`Self::mapping`] selects output channels.
    pub fn decoded_channels(&self) -> u16 {
        self.coupled_count as u16 + self.stream_count as u16
    }

    /// The output channel count `C` — the length of [`Self::mapping`].
    pub fn output_channels(&self) -> u8 {
        self.mapping.len() as u8
    }
}

/// A decoded `OpusHead` identification header (RFC 7845 §5.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpusHead {
    /// Raw version octet (RFC 7845 §5.1 item 2). The major version (upper
    /// nibble) has been validated ≤ [`OPUS_HEAD_MAX_VERSION`].
    pub version: u8,
    /// Output channel count `C` (§5.1 item 3); always ≥ 1.
    pub channel_count: u8,
    /// Pre-skip: 48 kHz samples to discard at start-up (§5.1 item 4).
    pub pre_skip: u16,
    /// Original input sample rate in Hz; 0 means "unspecified" (§5.1
    /// item 5). Metadata only — playback is always at 48 kHz here.
    pub input_sample_rate: u32,
    /// Output gain, Q7.8 dB, signed (§5.1 item 6). Apply via
    /// `pow(10, gain / (20.0 * 256))`.
    pub output_gain_q7_8: i16,
    /// Channel mapping family octet (§5.1 item 7).
    pub mapping_family: u8,
    /// The resolved §5.1.1 channel-mapping table (synthesized for
    /// family 0, parsed for families ≥ 1).
    pub mapping: ChannelMappingTable,
}

impl OpusHead {
    /// Parse and fully validate an identification-header packet per RFC
    /// 7845 §5.1 + §5.1.1.
    ///
    /// Enforces every MUST in the two sections: the magic signature, the
    /// version major-nibble bound, the non-zero channel count and its
    /// per-family range, the non-zero stream count, `M ≤ N`,
    /// `M + N ≤ 255`, and the per-output-channel mapping-index bound.
    pub fn parse(packet: &[u8]) -> Result<Self, OpusHeadError> {
        if packet.len() < OPUS_HEAD_MIN_LEN {
            return Err(OpusHeadError::TooShort {
                got: packet.len(),
                need: OPUS_HEAD_MIN_LEN,
            });
        }
        if &packet[0..8] != OPUS_HEAD_MAGIC.as_slice() {
            return Err(OpusHeadError::BadMagic);
        }
        let version = packet[8];
        if version >> 4 > 0 {
            return Err(OpusHeadError::IncompatibleVersion { version });
        }
        let channel_count = packet[9];
        if channel_count == 0 {
            return Err(OpusHeadError::ZeroChannels);
        }
        let pre_skip = u16::from_le_bytes([packet[10], packet[11]]);
        let input_sample_rate =
            u32::from_le_bytes([packet[12], packet[13], packet[14], packet[15]]);
        let output_gain_q7_8 = i16::from_le_bytes([packet[16], packet[17]]);
        let mapping_family = packet[18];

        let mapping = if mapping_family == 0 {
            // §5.1.1.1 family 0: 1 or 2 channels; the table is omitted
            // and synthesized from the pinned defaults (N = 1,
            // M = C − 1, identity map).
            if channel_count > 2 {
                return Err(OpusHeadError::ChannelCountForFamily {
                    family: mapping_family,
                    channels: channel_count,
                });
            }
            ChannelMappingTable {
                stream_count: 1,
                coupled_count: channel_count - 1,
                mapping: (0..channel_count).collect(),
            }
        } else {
            // §5.1.1 families ≥ 1 carry an explicit table: N (1 byte),
            // M (1 byte), then C mapping octets.
            // `channel_count` is already validated non-zero above, so
            // the family-1 range check is just the upper bound of 8.
            if mapping_family == 1 && channel_count > 8 {
                return Err(OpusHeadError::ChannelCountForFamily {
                    family: mapping_family,
                    channels: channel_count,
                });
            }
            let table_len = 2 + channel_count as usize;
            let need = OPUS_HEAD_MIN_LEN + table_len;
            if packet.len() < need {
                return Err(OpusHeadError::TooShort {
                    got: packet.len(),
                    need,
                });
            }
            let stream_count = packet[19];
            let coupled_count = packet[20];
            if stream_count == 0 {
                return Err(OpusHeadError::ZeroStreams);
            }
            if coupled_count > stream_count {
                return Err(OpusHeadError::CoupledExceedsStreams {
                    streams: stream_count,
                    coupled: coupled_count,
                });
            }
            // M + N ≤ 255 (the index space cannot address more).
            if coupled_count as u16 + stream_count as u16 > 255 {
                return Err(OpusHeadError::TooManyDecodedChannels {
                    streams: stream_count,
                    coupled: coupled_count,
                });
            }
            let decoded_channels = coupled_count + stream_count; // ≤ 255, fits u8.
            let map_start = 21;
            let mut mapping = Vec::with_capacity(channel_count as usize);
            for c in 0..channel_count as usize {
                let index = packet[map_start + c];
                if index != 255 && index >= decoded_channels {
                    return Err(OpusHeadError::MappingIndexOutOfRange {
                        output_channel: c as u8,
                        index,
                        decoded_channels,
                    });
                }
                mapping.push(index);
            }
            ChannelMappingTable {
                stream_count,
                coupled_count,
                mapping,
            }
        };

        Ok(OpusHead {
            version,
            channel_count,
            pre_skip,
            input_sample_rate,
            output_gain_q7_8,
            mapping_family,
            mapping,
        })
    }

    /// Compose the identification-header packet for this `OpusHead` —
    /// the write-side mirror of [`Self::parse`] (RFC 7845 §5.1 +
    /// §5.1.1).
    ///
    /// Validates the same MUSTs the parser enforces before emitting a
    /// single byte: the version major-nibble bound, the non-zero
    /// channel count and its per-family range, a mapping-table length
    /// equal to the channel count, the non-zero stream count, `M ≤ N`,
    /// `M + N ≤ 255`, and the per-output-channel mapping-index bound.
    /// For **family 0** the on-wire header omits the table (§5.1.1.1),
    /// so the held table must equal the RFC-pinned synthesized default
    /// (`N = 1`, `M = C − 1`, identity map) or composition fails —
    /// otherwise the parse of the produced bytes would not reconstruct
    /// this value. Successful output always reparses equal.
    pub fn compose(&self) -> Result<Vec<u8>, OpusHeadError> {
        if self.version >> 4 > 0 {
            return Err(OpusHeadError::IncompatibleVersion {
                version: self.version,
            });
        }
        if self.channel_count == 0 {
            return Err(OpusHeadError::ZeroChannels);
        }
        let mut out = Vec::with_capacity(OPUS_HEAD_MIN_LEN + 2 + self.channel_count as usize);
        out.extend_from_slice(OPUS_HEAD_MAGIC);
        out.push(self.version);
        out.push(self.channel_count);
        out.extend_from_slice(&self.pre_skip.to_le_bytes());
        out.extend_from_slice(&self.input_sample_rate.to_le_bytes());
        out.extend_from_slice(&self.output_gain_q7_8.to_le_bytes());
        out.push(self.mapping_family);

        if self.mapping_family == 0 {
            // §5.1.1.1: C ≤ 2, table omitted; the held table must be
            // exactly the synthesized default.
            if self.channel_count > 2 {
                return Err(OpusHeadError::ChannelCountForFamily {
                    family: 0,
                    channels: self.channel_count,
                });
            }
            let default = ChannelMappingTable {
                stream_count: 1,
                coupled_count: self.channel_count - 1,
                mapping: (0..self.channel_count).collect(),
            };
            if self.mapping != default {
                // A family-0 header cannot carry a non-default table;
                // surface it as the family/channel mismatch it is.
                return Err(OpusHeadError::ChannelCountForFamily {
                    family: 0,
                    channels: self.channel_count,
                });
            }
        } else {
            if self.mapping_family == 1 && self.channel_count > 8 {
                return Err(OpusHeadError::ChannelCountForFamily {
                    family: self.mapping_family,
                    channels: self.channel_count,
                });
            }
            if self.mapping.mapping.len() != self.channel_count as usize {
                return Err(OpusHeadError::TooShort {
                    got: OPUS_HEAD_MIN_LEN + 2 + self.mapping.mapping.len(),
                    need: OPUS_HEAD_MIN_LEN + 2 + self.channel_count as usize,
                });
            }
            if self.mapping.stream_count == 0 {
                return Err(OpusHeadError::ZeroStreams);
            }
            if self.mapping.coupled_count > self.mapping.stream_count {
                return Err(OpusHeadError::CoupledExceedsStreams {
                    streams: self.mapping.stream_count,
                    coupled: self.mapping.coupled_count,
                });
            }
            if self.mapping.decoded_channels() > 255 {
                return Err(OpusHeadError::TooManyDecodedChannels {
                    streams: self.mapping.stream_count,
                    coupled: self.mapping.coupled_count,
                });
            }
            // `M + N ≤ 255` was checked above, so the u8 sum cannot wrap.
            let decoded_channels = self.mapping.coupled_count + self.mapping.stream_count;
            out.push(self.mapping.stream_count);
            out.push(self.mapping.coupled_count);
            for (c, &index) in self.mapping.mapping.iter().enumerate() {
                if index != 255 && index >= decoded_channels {
                    return Err(OpusHeadError::MappingIndexOutOfRange {
                        output_channel: c as u8,
                        index,
                        decoded_channels,
                    });
                }
                out.push(index);
            }
        }
        Ok(out)
    }

    /// Linear playback scale factor derived from the §5.1 output gain:
    /// `pow(10, output_gain / (20.0 * 256))`. A gain of 0 returns 1.0.
    pub fn output_gain_linear(&self) -> f64 {
        10f64.powf(self.output_gain_q7_8 as f64 / (20.0 * 256.0))
    }

    /// Apply this header's §5.1 output gain in place to a buffer of
    /// interleaved 48 kHz PCM. A zero gain is a no-op; otherwise every
    /// sample is scaled by [`Self::output_gain_linear`] and saturated to
    /// the `i16` range. See [`apply_output_gain`].
    pub fn apply_gain(&self, pcm: &mut [i16]) {
        apply_output_gain(pcm, self.output_gain_q7_8);
    }
}

/// Apply a §5.1 output gain (raw Q7.8 dB value) in place to a buffer of
/// PCM samples, saturating to the `i16` range.
///
/// RFC 7845 §5.1 item 6 defines the gain as a Q7.8 fixed-point dB value
/// and gives the application formula
/// `sample *= pow(10, output_gain / (20.0 * 256))`. "Players and media
/// frameworks SHOULD apply it by default." A gain of 0 leaves the buffer
/// untouched (the common case — muxers SHOULD write zero and bake any
/// gain into the encode).
pub fn apply_output_gain(pcm: &mut [i16], gain_q7_8: i16) {
    if gain_q7_8 == 0 {
        return;
    }
    let scale = 10f64.powf(gain_q7_8 as f64 / (20.0 * 256.0));
    for s in pcm.iter_mut() {
        let scaled = (*s as f64 * scale).round();
        *s = scaled.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
    }
}

/// Pre-skip accumulator (RFC 7845 §5.1 item 4 / §4.2 trimming).
///
/// The §5.1 pre-skip is the number of 48 kHz samples (per channel) to
/// discard from the *start* of the decoded output, giving the decoder's
/// internal filters time to converge before audible playback begins.
/// This helper threads the remaining pre-skip count across packets: feed
/// it each decoded packet's per-channel sample count and it reports how
/// many leading per-channel samples of that packet to drop.
///
/// Trimming is applied to the per-channel sample stream; for interleaved
/// PCM with `C` channels, multiply the returned count by `C` to get the
/// number of interleaved samples to drop from the front of the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreSkip {
    remaining: u32,
}

impl PreSkip {
    /// Construct a pre-skip accumulator with `pre_skip` per-channel
    /// samples still to discard.
    pub fn new(pre_skip: u16) -> Self {
        PreSkip {
            remaining: pre_skip as u32,
        }
    }

    /// Build the pre-skip accumulator for a parsed [`OpusHead`].
    pub fn from_head(head: &OpusHead) -> Self {
        Self::new(head.pre_skip)
    }

    /// Per-channel samples still to discard.
    pub fn remaining(&self) -> u32 {
        self.remaining
    }

    /// `true` once the pre-skip region has been fully consumed.
    pub fn is_done(&self) -> bool {
        self.remaining == 0
    }

    /// Register `samples_per_channel` newly-decoded per-channel samples
    /// and return how many of them (from the front) fall inside the
    /// pre-skip region and must be discarded. The remaining count is
    /// advanced accordingly.
    pub fn consume(&mut self, samples_per_channel: usize) -> usize {
        let drop = (samples_per_channel as u32).min(self.remaining);
        self.remaining -= drop;
        drop as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal family-0 header with the given channel count.
    fn family0_header(channels: u8) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(OPUS_HEAD_MAGIC);
        h.push(1); // version
        h.push(channels);
        h.extend_from_slice(&3840u16.to_le_bytes()); // pre-skip
        h.extend_from_slice(&48000u32.to_le_bytes()); // input rate
        h.extend_from_slice(&0i16.to_le_bytes()); // gain
        h.push(0); // mapping family 0
        h
    }

    #[test]
    fn family0_mono_defaults() {
        let head = OpusHead::parse(&family0_header(1)).unwrap();
        assert_eq!(head.channel_count, 1);
        assert_eq!(head.pre_skip, 3840);
        assert_eq!(head.input_sample_rate, 48000);
        assert_eq!(head.mapping.stream_count, 1);
        assert_eq!(head.mapping.coupled_count, 0);
        assert_eq!(head.mapping.mapping, vec![0]);
        assert_eq!(head.mapping.decoded_channels(), 1);
    }

    #[test]
    fn family0_stereo_defaults() {
        let head = OpusHead::parse(&family0_header(2)).unwrap();
        assert_eq!(head.mapping.stream_count, 1);
        assert_eq!(head.mapping.coupled_count, 1); // C − 1
        assert_eq!(head.mapping.mapping, vec![0, 1]);
        assert_eq!(head.mapping.decoded_channels(), 2);
    }

    #[test]
    fn family0_rejects_more_than_two_channels() {
        assert_eq!(
            OpusHead::parse(&family0_header(3)),
            Err(OpusHeadError::ChannelCountForFamily {
                family: 0,
                channels: 3,
            })
        );
    }

    #[test]
    fn bad_magic_rejected() {
        let mut h = family0_header(1);
        h[0] = b'X';
        assert_eq!(OpusHead::parse(&h), Err(OpusHeadError::BadMagic));
    }

    #[test]
    fn too_short_rejected() {
        let h = vec![0u8; 10];
        assert_eq!(
            OpusHead::parse(&h),
            Err(OpusHeadError::TooShort { got: 10, need: 19 })
        );
    }

    #[test]
    fn incompatible_major_version_rejected() {
        let mut h = family0_header(1);
        h[8] = 0x10; // major version 1 ⇒ incompatible
        assert_eq!(
            OpusHead::parse(&h),
            Err(OpusHeadError::IncompatibleVersion { version: 0x10 })
        );
        // Minor-version bump within major 0 stays compatible.
        let mut h = family0_header(1);
        h[8] = 0x0F;
        assert!(OpusHead::parse(&h).is_ok());
    }

    #[test]
    fn zero_channels_rejected() {
        let mut h = family0_header(1);
        h[9] = 0;
        assert_eq!(OpusHead::parse(&h), Err(OpusHeadError::ZeroChannels));
    }

    /// A 6-channel (5.1) family-1 header: N = 4 streams, M = 2 coupled,
    /// Vorbis-order identity mapping `[0,1,2,3,4,5]`.
    fn family1_5_1_header() -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(OPUS_HEAD_MAGIC);
        h.push(1);
        h.push(6); // 6 output channels
        h.extend_from_slice(&312u16.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(&0i16.to_le_bytes());
        h.push(1); // mapping family 1
        h.push(4); // N
        h.push(2); // M
        h.extend_from_slice(&[0, 1, 2, 3, 4, 5]); // C mapping octets
        h
    }

    #[test]
    fn family1_surround_table() {
        let head = OpusHead::parse(&family1_5_1_header()).unwrap();
        assert_eq!(head.mapping_family, 1);
        assert_eq!(head.channel_count, 6);
        assert_eq!(head.mapping.stream_count, 4);
        assert_eq!(head.mapping.coupled_count, 2);
        assert_eq!(head.mapping.decoded_channels(), 6); // M + N
        assert_eq!(head.mapping.mapping, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(head.mapping.output_channels(), 6);
    }

    #[test]
    fn family1_rejects_coupled_gt_streams() {
        let mut h = family1_5_1_header();
        h[20] = 5; // M = 5 > N = 4
        assert_eq!(
            OpusHead::parse(&h),
            Err(OpusHeadError::CoupledExceedsStreams {
                streams: 4,
                coupled: 5,
            })
        );
    }

    #[test]
    fn family1_rejects_zero_streams() {
        let mut h = family1_5_1_header();
        h[19] = 0;
        assert_eq!(OpusHead::parse(&h), Err(OpusHeadError::ZeroStreams));
    }

    #[test]
    fn family1_rejects_out_of_range_mapping_index() {
        let mut h = family1_5_1_header();
        h[21 + 5] = 6; // M + N = 6, so index 6 is out of range (and not 255)
        assert_eq!(
            OpusHead::parse(&h),
            Err(OpusHeadError::MappingIndexOutOfRange {
                output_channel: 5,
                index: 6,
                decoded_channels: 6,
            })
        );
        // The reserved silence index 255 is always accepted.
        let mut h = family1_5_1_header();
        h[21 + 5] = 255;
        assert!(OpusHead::parse(&h).is_ok());
    }

    #[test]
    fn family1_rejects_truncated_table() {
        let mut h = family1_5_1_header();
        h.truncate(22); // cut into the mapping octets
        assert!(matches!(
            OpusHead::parse(&h),
            Err(OpusHeadError::TooShort { .. })
        ));
    }

    #[test]
    fn family1_channel_count_bounds() {
        // Family 1 allows 1..=8; 9 channels is rejected.
        let mut h = Vec::new();
        h.extend_from_slice(OPUS_HEAD_MAGIC);
        h.push(1);
        h.push(9);
        h.extend_from_slice(&0u16.to_le_bytes());
        h.extend_from_slice(&0u32.to_le_bytes());
        h.extend_from_slice(&0i16.to_le_bytes());
        h.push(1);
        h.push(6);
        h.push(2);
        h.extend_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 0]);
        assert_eq!(
            OpusHead::parse(&h),
            Err(OpusHeadError::ChannelCountForFamily {
                family: 1,
                channels: 9,
            })
        );
    }

    #[test]
    fn output_gain_linear_unity_at_zero() {
        let head = OpusHead::parse(&family0_header(1)).unwrap();
        assert!((head.output_gain_linear() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn output_gain_linear_positive_amplifies() {
        let mut h = family0_header(1);
        // +6.02 dB ≈ Q7.8 value 256*6 = 1536 → ~2x linear.
        let g: i16 = 1536;
        h[16..18].copy_from_slice(&g.to_le_bytes());
        let head = OpusHead::parse(&h).unwrap();
        assert!(head.output_gain_linear() > 1.9 && head.output_gain_linear() < 2.1);
    }

    #[test]
    fn apply_output_gain_zero_is_noop() {
        let mut pcm = vec![100i16, -200, 32767, -32768];
        let before = pcm.clone();
        apply_output_gain(&mut pcm, 0);
        assert_eq!(pcm, before);
    }

    #[test]
    fn apply_output_gain_doubles_at_plus_6db() {
        // +6.02 dB ≈ Q7.8 1536 → ×2.
        let mut pcm = vec![100i16, -100, 50];
        apply_output_gain(&mut pcm, 1536);
        assert_eq!(pcm[0], 200);
        assert_eq!(pcm[1], -200);
        assert_eq!(pcm[2], 100);
    }

    #[test]
    fn apply_output_gain_saturates() {
        // A large positive gain pushes a mid-scale sample past i16::MAX;
        // it must clamp, not wrap.
        let mut pcm = vec![20000i16, -20000];
        apply_output_gain(&mut pcm, 1536); // ×2 → 40000 > 32767
        assert_eq!(pcm[0], i16::MAX);
        assert_eq!(pcm[1], i16::MIN);
    }

    #[test]
    fn apply_gain_via_head() {
        let mut h = family0_header(1);
        h[16..18].copy_from_slice(&1536i16.to_le_bytes());
        let head = OpusHead::parse(&h).unwrap();
        let mut pcm = vec![10i16, -10];
        head.apply_gain(&mut pcm);
        assert_eq!(pcm, vec![20, -20]);
    }

    #[test]
    fn pre_skip_consumes_across_packets() {
        // 312-sample pre-skip (the fixture value) spread across 20 ms
        // packets of 960 samples each: the first packet drops all 312,
        // subsequent packets drop none.
        let mut ps = PreSkip::new(312);
        assert!(!ps.is_done());
        assert_eq!(ps.remaining(), 312);
        assert_eq!(ps.consume(960), 312);
        assert!(ps.is_done());
        assert_eq!(ps.consume(960), 0);
    }

    /// compose ∘ parse is the identity on header bytes: the family-0
    /// mono/stereo defaults and the family-1 5.1 table all re-emit
    /// byte-identically, and compose ∘ parse on a composed value
    /// roundtrips the struct.
    #[test]
    fn compose_roundtrips_parse() {
        for bytes in [family0_header(1), family0_header(2), family1_5_1_header()] {
            let head = OpusHead::parse(&bytes).unwrap();
            let composed = head.compose().unwrap();
            assert_eq!(composed, bytes);
            assert_eq!(OpusHead::parse(&composed).unwrap(), head);
        }
    }

    /// compose enforces the same MUSTs as parse: bad version nibble,
    /// zero channels, family-0 with a non-default table or > 2
    /// channels, zero streams, M > N, oversized index space, mapping
    /// length mismatch, and out-of-range mapping indices.
    #[test]
    fn compose_rejects_invalid_headers() {
        let base = OpusHead::parse(&family1_5_1_header()).unwrap();

        let mut h = base.clone();
        h.version = 0x20;
        assert!(matches!(
            h.compose(),
            Err(OpusHeadError::IncompatibleVersion { .. })
        ));

        let mut h = base.clone();
        h.channel_count = 0;
        assert_eq!(h.compose(), Err(OpusHeadError::ZeroChannels));

        // Family 0 with a non-default table.
        let mut h = OpusHead::parse(&family0_header(2)).unwrap();
        h.mapping.mapping = vec![1, 0];
        assert!(matches!(
            h.compose(),
            Err(OpusHeadError::ChannelCountForFamily { family: 0, .. })
        ));

        // Family 0 cannot carry more than 2 channels.
        let mut h = OpusHead::parse(&family0_header(2)).unwrap();
        h.channel_count = 3;
        h.mapping.mapping = vec![0, 1, 2];
        assert!(matches!(
            h.compose(),
            Err(OpusHeadError::ChannelCountForFamily { family: 0, .. })
        ));

        let mut h = base.clone();
        h.mapping.stream_count = 0;
        h.mapping.coupled_count = 0;
        assert_eq!(h.compose(), Err(OpusHeadError::ZeroStreams));

        let mut h = base.clone();
        h.mapping.coupled_count = h.mapping.stream_count + 1;
        assert!(matches!(
            h.compose(),
            Err(OpusHeadError::CoupledExceedsStreams { .. })
        ));

        let mut h = base.clone();
        h.mapping.stream_count = 200;
        h.mapping.coupled_count = 100;
        assert!(matches!(
            h.compose(),
            Err(OpusHeadError::TooManyDecodedChannels { .. })
        ));

        // Mapping-table length must equal the channel count.
        let mut h = base.clone();
        h.mapping.mapping.pop();
        assert!(matches!(h.compose(), Err(OpusHeadError::TooShort { .. })));

        // Mapping index outside the decoded-channel space (and != 255).
        let mut h = base.clone();
        h.mapping.mapping[3] = 6; // decoded channels = M + N = 6 ⇒ max 5
        assert!(matches!(
            h.compose(),
            Err(OpusHeadError::MappingIndexOutOfRange { .. })
        ));
        // ... while 255 (silence) is legal.
        let mut h = base.clone();
        h.mapping.mapping[3] = 255;
        let bytes = h.compose().unwrap();
        assert_eq!(OpusHead::parse(&bytes).unwrap(), h);
    }

    #[test]
    fn pre_skip_spanning_multiple_packets() {
        // A pre-skip larger than one packet drains over several.
        let mut ps = PreSkip::new(1500);
        assert_eq!(ps.consume(960), 960);
        assert_eq!(ps.remaining(), 540);
        assert_eq!(ps.consume(960), 540);
        assert!(ps.is_done());
        assert_eq!(ps.consume(960), 0);
    }

    #[test]
    fn pre_skip_from_head() {
        let head = OpusHead::parse(&family0_header(1)).unwrap();
        // family0_header sets pre-skip 3840.
        let ps = PreSkip::from_head(&head);
        assert_eq!(ps.remaining(), 3840);
    }
}
