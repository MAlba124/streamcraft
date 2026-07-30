//! CELT-only Opus **packet** encoder — §3.1 TOC + one §4.3 CELT frame
//! per packet (code 0), at a caller-chosen constant payload size
//! (RFC 6716 §3 / §4.3 / §5.3).
//!
//! The packets decode end-to-end through the crate's own
//! [`crate::decoder::OpusDecoder`]; the encode→decode chain carries
//! the fixed 2.5 ms §4.3.7 MDCT-overlap delay
//! ([`crate::celt_analysis`]).
//!
//! ## Provenance
//!
//! RFC 6716 §3 / §5.3 + the normative Appendix A reference listing
//! (staged `docs/audio/opus/rfc6716-opus.txt`, hash-verified per
//! §A.1). No external library source was consulted.

use crate::celt_frame_encode::{encode_celt_frame, CeltEncoderState, CeltFrameEncodeInfo};
use crate::range_encoder::RangeEncoder;
use crate::toc::{Bandwidth, FrameCountCode, Mode, OpusTocByte};
use crate::Error;

/// The §3.2 maximum Opus frame payload.
const MAX_FRAME_BYTES: usize = 1275;

/// A CELT-only packet encoder for one stream configuration.
#[derive(Debug, Clone)]
pub struct CeltEncoder {
    state: CeltEncoderState,
    bandwidth: Bandwidth,
    frame_tenths_ms: u16,
    stereo: bool,
    end_band: usize,
    lm: i32,
}

impl CeltEncoder {
    /// New CELT-only encoder. `bandwidth` selects the coded band range
    /// (NB→13, WB→17, SWB→19, FB→21; MB is not a CELT bandwidth) and
    /// `frame_tenths_ms` the frame duration (25/50/100/200 tenths of
    /// a millisecond).
    pub fn new(bandwidth: Bandwidth, frame_tenths_ms: u16, stereo: bool) -> Result<Self, Error> {
        let end_band = match bandwidth {
            Bandwidth::Nb => 13,
            Bandwidth::Wb => 17,
            Bandwidth::Swb => 19,
            Bandwidth::Fb => 21,
            Bandwidth::Mb => return Err(Error::MalformedPacket),
        };
        let lm = match frame_tenths_ms {
            25 => 0i32,
            50 => 1,
            100 => 2,
            200 => 3,
            _ => return Err(Error::MalformedPacket),
        };
        // Validate the TOC row exists up front.
        let _ = OpusTocByte::compose_byte(
            Mode::CeltOnly,
            bandwidth,
            frame_tenths_ms,
            stereo,
            FrameCountCode::One,
        )?;
        let n = 120usize << lm;
        let channels = if stereo { 2 } else { 1 };
        Ok(Self {
            state: CeltEncoderState::new(channels, n),
            bandwidth,
            frame_tenths_ms,
            stereo,
            end_band,
            lm,
        })
    }

    /// Samples per channel consumed by one packet (48 kHz).
    #[must_use]
    pub fn frame_samples(&self) -> usize {
        self.state.frame_len()
    }

    /// Channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.state.channels()
    }

    /// Reset all carried state (stream start / §4.5.2).
    pub fn reset(&mut self) {
        self.state.reset();
    }

    /// Encode one frame of interleaved 48 kHz PCM
    /// (`channels * frame_samples()` values) into a code-0 Opus packet
    /// of exactly `1 + payload_bytes` bytes.
    ///
    /// `payload_bytes` is the CELT frame budget (2..=1275); a constant
    /// value gives CBR transport.
    pub fn encode_packet(
        &mut self,
        pcm: &[i16],
        payload_bytes: usize,
    ) -> Result<(Vec<u8>, CeltFrameEncodeInfo), Error> {
        if pcm.len() != self.channels() * self.frame_samples() {
            return Err(Error::MalformedPacket);
        }
        if !(2..=MAX_FRAME_BYTES).contains(&payload_bytes) {
            return Err(Error::MalformedPacket);
        }
        let toc = OpusTocByte::compose_byte(
            Mode::CeltOnly,
            self.bandwidth,
            self.frame_tenths_ms,
            self.stereo,
            FrameCountCode::One,
        )?;
        let mut enc = RangeEncoder::new();
        let info = encode_celt_frame(
            &mut self.state,
            &mut enc,
            pcm,
            payload_bytes,
            0,
            self.end_band,
            self.lm,
        );
        debug_assert!(enc.tell() as usize <= payload_bytes * 8, "budget bust");
        let payload = enc
            .finish_fixed(payload_bytes)
            .ok_or(Error::MalformedPacket)?;
        let mut packet = Vec::with_capacity(1 + payload_bytes);
        packet.push(toc);
        packet.extend_from_slice(&payload);
        Ok((packet, info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_configs() {
        assert!(CeltEncoder::new(Bandwidth::Mb, 200, false).is_err());
        assert!(CeltEncoder::new(Bandwidth::Fb, 400, false).is_err());
        let mut e = CeltEncoder::new(Bandwidth::Fb, 200, false).unwrap();
        assert_eq!(e.frame_samples(), 960);
        let pcm = vec![0i16; 960];
        assert!(e.encode_packet(&pcm, 1).is_err());
        assert!(e.encode_packet(&pcm[..100], 100).is_err());
    }

    #[test]
    fn digital_silence_encodes_and_decodes_as_celt_silence() {
        let mut e = CeltEncoder::new(Bandwidth::Fb, 200, false).unwrap();
        let pcm = vec![0i16; 960];
        let (packet, info) = e.encode_packet(&pcm, 60).unwrap();
        assert!(info.silence);
        assert_eq!(packet.len(), 61);
        let mut dec = crate::decoder::OpusDecoder::new();
        let out = dec.decode_packet(&packet).unwrap();
        assert_eq!(out.samples_per_channel(), 960);
        assert!(out.pcm.iter().all(|&v| v == 0));
    }
}
