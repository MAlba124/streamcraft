//! A FLAC decoder sufficient to decode what [`FlacEncoder`](crate::FlacEncoder)
//! emits (spec: RFC 9639 §8, §9; `spec/rfc9639.txt`). Its purpose is to *prove*
//! round-trip losslessness in-crate, so it implements exactly the feature set the
//! encoder produces: STREAMINFO, fixed-block-size frames, and the `CONSTANT`,
//! `VERBATIM`, and `FIXED` subframe types with partitioned Rice residuals.
//!
//! Stereo decorrelation (§4.2) and LPC subframes (§9.2.6) are parsed defensively —
//! LPC and the escape residual method are recognised and either handled or reported
//! as an error rather than mis-decoded — but the encoder never emits them, so the
//! round-trip tests exercise the common path. As a decoder parses untrusted input,
//! every read is checked and any malformed bitstream yields a [`DecodeError`], never
//! a panic (spec: decoders parse untrusted input — a crash is a P0).

use crate::bitstream::{crc16, crc8, BitReader, ReadError};

/// The STREAMINFO fields a decoder needs (§8.2 Table 3). MD5 is carried verbatim but
/// not verified here (the encoder writes zeros = "unknown").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamInfo {
    pub min_block_size: u32,
    pub max_block_size: u32,
    pub min_frame_size: u32,
    pub max_frame_size: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub bits_per_sample: u32,
    pub total_samples: u64,
    pub md5: [u8; 16],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Missing or wrong `fLaC` marker.
    BadMagic,
    /// First metadata block was not STREAMINFO, or its length was wrong.
    BadStreamInfo,
    /// A frame sync code did not match 0b111111111111100.
    BadSync,
    /// A reserved / forbidden field value was encountered.
    Reserved(&'static str),
    /// A CRC-8 (header) or CRC-16 (frame) check failed.
    CrcMismatch,
    /// A feature the encoder never emits and this decoder does not implement.
    Unsupported(&'static str),
    /// Ran off the end of the input.
    UnexpectedEof,
    /// Internal invariant broken (would indicate an encoder/decoder disagreement).
    Corrupt(&'static str),
}

impl From<ReadError> for DecodeError {
    fn from(_: ReadError) -> Self {
        DecodeError::UnexpectedEof
    }
}

/// A fully decoded FLAC stream: the header plus the reconstructed interleaved PCM as
/// per-sample i64 (sign-extended). Interleaving is channel-major within each
/// interchannel sample, matching the encoder's input order.
pub struct FlacDecoder {
    pub info: StreamInfo,
    /// Decoded interchannel samples, interleaved: `[s0c0, s0c1, ..., s1c0, ...]`.
    pub samples: Vec<i64>,
}

impl FlacDecoder {
    /// Decode an entire in-memory FLAC stream.
    pub fn decode(data: &[u8]) -> Result<Self, DecodeError> {
        let mut r = BitReader::new(data);

        // fLaC marker (§6): 0x664C6143.
        let magic = r.read_bits(32)?;
        if magic != 0x664C_6143 {
            return Err(DecodeError::BadMagic);
        }

        let info = Self::read_metadata(&mut r)?;

        let mut samples: Vec<i64> = Vec::new();
        // Frames follow until the input is exhausted (§9). A fixed-block-size stream
        // has no explicit frame count; we stop when fewer than a sync code's worth of
        // bits remain.
        while r.bits_left() >= 16 {
            Self::read_frame(&mut r, data, &info, &mut samples)?;
            r.align(); // frames are byte-aligned (§9.3 footer padding)
        }
        Ok(Self { info, samples })
    }

    /// Read the metadata block chain and return the STREAMINFO (§8.1, §8.2). Non-
    /// STREAMINFO blocks are skipped by their declared length.
    fn read_metadata(r: &mut BitReader) -> Result<StreamInfo, DecodeError> {
        // First block MUST be STREAMINFO (§8).
        let last_first = r.read_bits(1)?;
        let block_type = r.read_bits(7)?;
        let length = r.read_bits(24)? as usize;
        if block_type != 0 || length != 34 {
            return Err(DecodeError::BadStreamInfo);
        }
        let info = Self::read_streaminfo(r)?;

        let mut last = last_first == 1;
        while !last {
            last = r.read_bits(1)? == 1;
            let _typ = r.read_bits(7)?;
            let len = r.read_bits(24)? as usize;
            // Skip the block body.
            r.read_aligned_bytes(len)?;
        }
        Ok(info)
    }

    /// STREAMINFO body (§8.2 Table 3).
    fn read_streaminfo(r: &mut BitReader) -> Result<StreamInfo, DecodeError> {
        let min_block_size = r.read_bits(16)? as u32;
        let max_block_size = r.read_bits(16)? as u32;
        let min_frame_size = r.read_bits(24)? as u32;
        let max_frame_size = r.read_bits(24)? as u32;
        let sample_rate = r.read_bits(20)? as u32;
        let channels = r.read_bits(3)? as u32 + 1;
        let bits_per_sample = r.read_bits(5)? as u32 + 1;
        let total_samples = r.read_bits(36)?;
        let mut md5 = [0u8; 16];
        for b in md5.iter_mut() {
            *b = r.read_bits(8)? as u8;
        }
        Ok(StreamInfo {
            min_block_size,
            max_block_size,
            min_frame_size,
            max_frame_size,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples,
            md5,
        })
    }

    /// Decode one frame, appending its interleaved samples to `out` (§9.1–§9.3).
    /// `data` is the whole stream, used to CRC the frame's byte range.
    fn read_frame(
        r: &mut BitReader,
        data: &[u8],
        info: &StreamInfo,
        out: &mut Vec<i64>,
    ) -> Result<(), DecodeError> {
        debug_assert!(r.is_byte_aligned());
        let frame_start_byte = (r.bit_pos() / 8) as usize;

        // Frame header (§9.1).
        let sync = r.read_bits(15)?;
        if sync != 0b111111111111100 {
            return Err(DecodeError::BadSync);
        }
        let blocking_strategy = r.read_bits(1)?; // 0 fixed, 1 variable
        let bs_bits = r.read_bits(4)? as u32;
        let sr_bits = r.read_bits(4)? as u32;
        let ch_bits = r.read_bits(4)? as u32;
        let bd_bits = r.read_bits(3)? as u32;
        let _reserved = r.read_bits(1)?; // MUST be 0

        // Coded number (§9.1.5). Value is unused for decoding but must be consumed.
        let _coded = read_coded_number(r)?;

        // Uncommon block size (§9.1.6) then uncommon sample rate (§9.1.7).
        let block_size = decode_block_size(r, bs_bits)?;
        let _rate = decode_sample_rate(r, sr_bits, info)?;
        let (channels, decorrelation) = decode_channels(ch_bits)?;
        let bit_depth = decode_bit_depth(bd_bits, info)?;
        let _ = blocking_strategy;

        // Header CRC-8 (§9.1.8): covers the header bytes just read (byte-aligned).
        let header_end_byte = (r.bit_pos() / 8) as usize;
        let crc_read = r.read_bits(8)? as u8;
        let header_bytes = data
            .get(frame_start_byte..header_end_byte)
            .ok_or(DecodeError::UnexpectedEof)?;
        if crc8(header_bytes) != crc_read {
            return Err(DecodeError::CrcMismatch);
        }

        // Subframes (§9.2): one per channel. Side channels carry an extra bit (§4.2).
        let mut planar: Vec<Vec<i64>> = Vec::with_capacity(channels as usize);
        for c in 0..channels {
            let extra = decorrelation.extra_bits_for_channel(c);
            let sub = read_subframe(r, block_size as usize, bit_depth + extra)?;
            planar.push(sub);
        }

        // Frame footer (§9.3): pad to byte boundary, then CRC-16 over the whole frame.
        r.align();
        let footer_end_byte = (r.bit_pos() / 8) as usize;
        let crc16_read = r.read_bits(16)? as u16;
        let frame_bytes = data
            .get(frame_start_byte..footer_end_byte)
            .ok_or(DecodeError::UnexpectedEof)?;
        if crc16(frame_bytes) != crc16_read {
            return Err(DecodeError::CrcMismatch);
        }

        // Undo stereo decorrelation (§4.2), then interleave channel-major.
        decorrelation.restore(&mut planar);
        for i in 0..block_size as usize {
            for c in 0..channels as usize {
                out.push(planar[c][i]);
            }
        }
        Ok(())
    }
}

/// One decoded FLAC frame: interleaved (channel-major) interchannel samples, sign-extended
/// to `i64` — the same order and representation as [`FlacDecoder::samples`].
pub struct DecodedFrame {
    pub samples: Vec<i64>,
}

/// An **incremental** FLAC decoder: push bytes as they arrive, pull decoded frames as they
/// complete (spec: decoders must not buffer the whole stream — latency #2). Only the
/// undecoded tail is retained — the consumed prefix is dropped — so memory stays on the
/// order of one frame regardless of stream length. It reuses [`FlacDecoder`]'s frame and
/// metadata readers, so the two decoders can never diverge.
#[derive(Default)]
pub struct StreamDecoder {
    buf: Vec<u8>,
    /// Decode cursor within `buf`; bytes before it are consumed and periodically dropped.
    pos: usize,
    info: Option<StreamInfo>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// The stream header, available once enough bytes have been pushed to parse it.
    pub fn info(&self) -> Option<&StreamInfo> {
        self.info.as_ref()
    }

    /// Append freshly-received bytes to the decode buffer.
    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Decode the next fully-buffered frame, or `Ok(None)` when more bytes are needed
    /// (a truncated frame is not an error — it is backpressure). The `fLaC` marker and
    /// STREAMINFO are parsed on the first call(s) with enough bytes; thereafter
    /// [`info`](Self::info) is populated. Malformed input returns `Err`, never a panic.
    pub fn pull(&mut self) -> Result<Option<DecodedFrame>, DecodeError> {
        if self.info.is_none() && !self.try_header()? {
            return Ok(None);
        }
        let info = self.info.expect("header parsed above");

        // Attempt one frame from the cursor. `read_frame` runs off the end of a truncated
        // slice as `UnexpectedEof`, which here means "need more" rather than corruption.
        let outcome = {
            let slice = &self.buf[self.pos..];
            if slice.len() < 2 {
                return Ok(None); // not even a frame sync yet
            }
            let mut r = BitReader::new(slice);
            let mut out = Vec::new();
            match FlacDecoder::read_frame(&mut r, slice, &info, &mut out) {
                Ok(()) => Some(((r.bit_pos() / 8) as usize, out)),
                Err(DecodeError::UnexpectedEof) => None,
                Err(e) => return Err(e),
            }
        };
        match outcome {
            Some((used, samples)) => {
                self.pos += used;
                self.compact();
                Ok(Some(DecodedFrame { samples }))
            }
            None => Ok(None),
        }
    }

    /// Parse the `fLaC` marker and metadata chain once enough bytes are buffered. Returns
    /// `Ok(false)` (need more) if the header is still truncated.
    fn try_header(&mut self) -> Result<bool, DecodeError> {
        let (info, used) = {
            let slice = &self.buf[self.pos..];
            let mut r = BitReader::new(slice);
            let magic = match r.read_bits(32) {
                Ok(m) => m,
                Err(_) => return Ok(false), // marker not fully here yet
            };
            if magic != 0x664C_6143 {
                return Err(DecodeError::BadMagic);
            }
            match FlacDecoder::read_metadata(&mut r) {
                Ok(info) => (info, (r.bit_pos() / 8) as usize),
                Err(DecodeError::UnexpectedEof) => return Ok(false), // metadata truncated
                Err(e) => return Err(e),
            }
        };
        self.info = Some(info);
        self.pos += used;
        self.compact();
        Ok(true)
    }

    /// Drop the consumed prefix once it grows past a threshold, so the buffer stays around
    /// one frame rather than the whole stream (the point of incremental decoding).
    fn compact(&mut self) {
        const DRAIN_AT: usize = 64 * 1024;
        if self.pos >= DRAIN_AT {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
    }
}

/// How the two channels of a stereo frame are correlated (§4.2, §9.1.3). Independent
/// channels need no restoration; left/side, side/right, and mid/side each reconstruct
/// left and right from the stored pair.
#[derive(Clone, Copy)]
enum Decorrelation {
    Independent,
    LeftSide,
    SideRight,
    MidSide,
}

impl Decorrelation {
    /// Extra bit-depth bits for subframe `c` (§4.2: the side channel is +1 bit).
    fn extra_bits_for_channel(self, c: u32) -> u32 {
        match self {
            Decorrelation::Independent => 0,
            Decorrelation::LeftSide => if c == 1 { 1 } else { 0 }, // side is right subframe
            Decorrelation::SideRight => if c == 0 { 1 } else { 0 }, // side is left subframe
            Decorrelation::MidSide => if c == 1 { 1 } else { 0 },   // side is second subframe
        }
    }

    /// Reconstruct left/right in place from the decorrelated pair (§4.2).
    fn restore(self, planar: &mut [Vec<i64>]) {
        match self {
            Decorrelation::Independent => {}
            Decorrelation::LeftSide => {
                // ch0 = left, ch1 = side = left - right => right = left - side.
                for i in 0..planar[0].len() {
                    let l = planar[0][i];
                    let s = planar[1][i];
                    planar[1][i] = l - s;
                }
            }
            Decorrelation::SideRight => {
                // ch0 = side = left - right, ch1 = right => left = side + right.
                for i in 0..planar[0].len() {
                    let s = planar[0][i];
                    let rgt = planar[1][i];
                    planar[0][i] = s + rgt;
                }
            }
            Decorrelation::MidSide => {
                // ch0 = mid = (left+right)>>1, ch1 = side = left-right.
                for i in 0..planar[0].len() {
                    let mid = planar[0][i];
                    let side = planar[1][i];
                    // Recover: mid was floor((l+r)/2); l+r = 2*mid + (side & 1).
                    let sum = (mid << 1) | (side & 1);
                    let l = (sum + side) >> 1;
                    let rgt = (sum - side) >> 1;
                    planar[0][i] = l;
                    planar[1][i] = rgt;
                }
            }
        }
    }
}

/// Decode one subframe of `block_size` samples at `bps` bit depth (§9.2).
fn read_subframe(r: &mut BitReader, block_size: usize, bps: u32) -> Result<Vec<i64>, DecodeError> {
    // Subframe header (§9.2.1): leading 0 bit, 6 type bits, wasted-bits flag.
    let zero = r.read_bits(1)?;
    if zero != 0 {
        return Err(DecodeError::Reserved("subframe leading bit must be 0"));
    }
    let type_bits = r.read_bits(6)? as u32;
    let wasted_flag = r.read_bits(1)?;
    let wasted = if wasted_flag == 1 {
        // Number of wasted bits minus 1, unary (§9.2.2).
        r.read_unary()? + 1
    } else {
        0
    };
    let effective_bps = bps
        .checked_sub(wasted)
        .filter(|&b| b > 0)
        .ok_or(DecodeError::Corrupt("wasted bits exceed bit depth"))?;

    let mut out = match type_bits {
        0b000000 => read_constant(r, block_size, effective_bps)?,
        0b000001 => read_verbatim(r, block_size, effective_bps)?,
        v if (0b001000..=0b001100).contains(&v) => {
            let order = v - 0b001000;
            read_fixed(r, block_size, effective_bps, order)?
        }
        v if v >= 0b100000 => {
            let order = v - 0b100000 + 1;
            read_lpc(r, block_size, effective_bps, order)?
        }
        _ => return Err(DecodeError::Reserved("reserved subframe type")),
    };

    // Undo wasted bits: shift decoded samples left by `wasted` (§9.2.2).
    if wasted > 0 {
        for s in out.iter_mut() {
            *s <<= wasted;
        }
    }
    Ok(out)
}

/// CONSTANT subframe: one sample repeated (§9.2.3).
fn read_constant(r: &mut BitReader, block_size: usize, bps: u32) -> Result<Vec<i64>, DecodeError> {
    let v = r.read_signed(bps)?;
    Ok(vec![v; block_size])
}

/// VERBATIM subframe: every sample stored unencoded (§9.2.4).
fn read_verbatim(r: &mut BitReader, block_size: usize, bps: u32) -> Result<Vec<i64>, DecodeError> {
    let mut out = Vec::with_capacity(block_size);
    for _ in 0..block_size {
        out.push(r.read_signed(bps)?);
    }
    Ok(out)
}

/// FIXED-predictor subframe of the given order (§9.2.5): warm-up samples, then the
/// coded residual, reconstructed by running the predictor forward.
fn read_fixed(
    r: &mut BitReader,
    block_size: usize,
    bps: u32,
    order: u32,
) -> Result<Vec<i64>, DecodeError> {
    let mut out = Vec::with_capacity(block_size);
    for _ in 0..order as usize {
        out.push(r.read_signed(bps)?); // unencoded warm-up (§9.2.5 Table 21)
    }
    let residual = read_residual(r, block_size, order)?;
    reconstruct_fixed(&mut out, &residual, order, block_size);
    Ok(out)
}

/// LPC subframe (§9.2.6). The encoder never emits LPC, but decoding it keeps the
/// decoder honest against real-world files used in cross-validation.
fn read_lpc(
    r: &mut BitReader,
    block_size: usize,
    bps: u32,
    order: u32,
) -> Result<Vec<i64>, DecodeError> {
    if order == 0 || order > 32 {
        return Err(DecodeError::Reserved("LPC order out of range"));
    }
    let mut out = Vec::with_capacity(block_size);
    for _ in 0..order as usize {
        out.push(r.read_signed(bps)?);
    }
    let precision = r.read_bits(4)? as u32 + 1; // (precision in bits) - 1 (§9.2.6 Table 22)
    if precision == 16 {
        return Err(DecodeError::Reserved("LPC precision 0b1111 is forbidden"));
    }
    let shift = r.read_signed(5)?;
    if shift < 0 {
        return Err(DecodeError::Reserved("LPC shift must not be negative"));
    }
    let mut coeffs = Vec::with_capacity(order as usize);
    for _ in 0..order {
        coeffs.push(r.read_signed(precision)?);
    }
    let residual = read_residual(r, block_size, order)?;
    for i in order as usize..block_size {
        let mut pred: i64 = 0;
        for j in 0..order as usize {
            pred += coeffs[j] * out[i - 1 - j];
        }
        pred >>= shift;
        out.push(pred + residual[i]);
    }
    // `out` already has warm-up + reconstructed; residual indices align by position.
    // (The loop above pushed the reconstructed samples; warm-up was pushed earlier.)
    debug_assert_eq!(out.len(), block_size);
    Ok(out)
}

/// Read a coded residual into a full-length vector (indices `0..order` are unused
/// placeholders; residual proper starts at `order`) (§9.2.7).
fn read_residual(r: &mut BitReader, block_size: usize, order: u32) -> Result<Vec<i64>, DecodeError> {
    let method = r.read_bits(2)?;
    let param_bits = match method {
        0b00 => 4,
        0b01 => 5,
        _ => return Err(DecodeError::Reserved("reserved residual coding method")),
    };
    let partition_order = r.read_bits(4)? as u32;
    let parts = 1usize << partition_order;
    if block_size % parts != 0 {
        return Err(DecodeError::Corrupt("block size not divisible by partitions"));
    }
    let part_len = block_size / parts;
    if part_len <= order as usize {
        return Err(DecodeError::Corrupt("partition smaller than predictor order"));
    }

    let mut residual = vec![0i64; block_size];
    let escape = (1u64 << param_bits) - 1; // 0b1111 or 0b11111 (§9.2.7)
    let mut idx = order as usize;
    for p in 0..parts {
        let count = if p == 0 { part_len - order as usize } else { part_len };
        let param = r.read_bits(param_bits)?;
        if param == escape {
            // Escaped partition (§9.2.7.1): 5-bit residual sample bit width, then raw.
            let width = r.read_bits(5)? as u32;
            for _ in 0..count {
                residual[idx] = if width == 0 { 0 } else { r.read_signed(width)? };
                idx += 1;
            }
        } else {
            let k = param as u32;
            for _ in 0..count {
                let q = r.read_unary()?; // most-significant part, unary (§9.2.7.2)
                let low = if k > 0 { r.read_bits(k)? } else { 0 };
                let folded = ((q as u64) << k) | low;
                residual[idx] = unzigzag(folded);
                idx += 1;
            }
        }
    }
    Ok(residual)
}

/// Run the FIXED predictor forward to reconstruct samples from residuals (§9.2.5).
/// `out` already holds `order` warm-up samples; residual holds the rest at matching
/// indices.
fn reconstruct_fixed(out: &mut Vec<i64>, residual: &[i64], order: u32, block_size: usize) {
    let o = order as usize;
    for i in o..block_size {
        let pred = match order {
            0 => 0,
            1 => out[i - 1],
            2 => 2 * out[i - 1] - out[i - 2],
            3 => 3 * out[i - 1] - 3 * out[i - 2] + out[i - 3],
            4 => 4 * out[i - 1] - 6 * out[i - 2] + 4 * out[i - 3] - out[i - 4],
            _ => unreachable!(),
        };
        out.push(pred + residual[i]);
    }
}

/// Inverse of the encoder's zigzag fold (§9.2.7.2): even -> n/2, odd -> -(n+1)/2.
#[inline]
fn unzigzag(folded: u64) -> i64 {
    ((folded >> 1) as i64) ^ -((folded & 1) as i64)
}

// --- Frame header field decoders (§9.1) ------------------------------------

/// Decode block size from the 4-bit code plus any uncommon extension (§9.1.1, §9.1.6).
fn decode_block_size(r: &mut BitReader, bs_bits: u32) -> Result<u32, DecodeError> {
    Ok(match bs_bits {
        0b0000 => return Err(DecodeError::Reserved("block size 0b0000 reserved")),
        0b0001 => 192,
        v @ 0b0010..=0b0101 => 144 * (1u32 << v),
        0b0110 => r.read_bits(8)? as u32 + 1, // uncommon, 8-bit
        0b0111 => r.read_bits(16)? as u32 + 1, // uncommon, 16-bit
        v => 1u32 << v, // 0b1000..0b1111 -> 2^v
    })
}

/// Decode sample rate from the 4-bit code plus any uncommon extension (§9.1.2,
/// §9.1.7). Falls back to STREAMINFO for code 0b0000.
fn decode_sample_rate(r: &mut BitReader, sr_bits: u32, info: &StreamInfo) -> Result<u32, DecodeError> {
    Ok(match sr_bits {
        0b0000 => info.sample_rate,
        0b0001 => 88200,
        0b0010 => 176400,
        0b0011 => 192000,
        0b0100 => 8000,
        0b0101 => 16000,
        0b0110 => 22050,
        0b0111 => 24000,
        0b1000 => 32000,
        0b1001 => 44100,
        0b1010 => 48000,
        0b1011 => 96000,
        0b1100 => r.read_bits(8)? as u32 * 1000, // kHz
        0b1101 => r.read_bits(16)? as u32,        // Hz
        0b1110 => r.read_bits(16)? as u32 * 10,   // Hz / 10
        _ => return Err(DecodeError::Reserved("sample rate 0b1111 forbidden")),
    })
}

/// Decode channel count and stereo decorrelation from the 4-bit channels code
/// (§9.1.3 Table 16).
fn decode_channels(ch_bits: u32) -> Result<(u32, Decorrelation), DecodeError> {
    Ok(match ch_bits {
        0b0000..=0b0111 => (ch_bits + 1, Decorrelation::Independent),
        0b1000 => (2, Decorrelation::LeftSide),
        0b1001 => (2, Decorrelation::SideRight),
        0b1010 => (2, Decorrelation::MidSide),
        _ => return Err(DecodeError::Reserved("reserved channel assignment")),
    })
}

/// Decode bit depth from the 3-bit code (§9.1.4 Table 17). Falls back to STREAMINFO
/// for code 0b000.
fn decode_bit_depth(bd_bits: u32, info: &StreamInfo) -> Result<u32, DecodeError> {
    Ok(match bd_bits {
        0b000 => info.bits_per_sample,
        0b001 => 8,
        0b010 => 12,
        0b011 => return Err(DecodeError::Reserved("bit depth 0b011 reserved")),
        0b100 => 16,
        0b101 => 20,
        0b110 => 24,
        0b111 => 32,
        _ => unreachable!("3-bit value"),
    })
}

/// Decode a UTF-8-like coded number (§9.1.5 Table 18), the inverse of the encoder's
/// `write_coded_number`. Public within the crate so encoder tests can round-trip it.
pub(crate) fn read_coded_number(r: &mut BitReader) -> Result<u64, DecodeError> {
    let first = r.read_bits(8)?;
    if first & 0x80 == 0 {
        return Ok(first); // 0xxxxxxx
    }
    // Count leading ones in the lead byte to get the total byte length.
    let mut mask = 0x80u64;
    let mut cont = 0u32;
    while first & mask != 0 {
        cont += 1;
        mask >>= 1;
    }
    // `cont` == number of leading ones == total bytes; continuation bytes = cont - 1.
    if !(2..=7).contains(&cont) {
        return Err(DecodeError::Reserved("invalid coded-number lead byte"));
    }
    let lead_bits = 8 - (cont + 1); // bits of value in the lead byte after the 0 separator
    let mut value = first & ((1u64 << lead_bits) - 1);
    for _ in 0..(cont - 1) {
        let byte = r.read_bits(8)?;
        if byte & 0xC0 != 0x80 {
            return Err(DecodeError::Reserved("bad coded-number continuation byte"));
        }
        value = (value << 6) | (byte & 0x3F);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unzigzag_inverts_zigzag() {
        for n in [-5i64, -2, -1, 0, 1, 2, 19, 1000, -1000] {
            let folded = ((n << 1) ^ (n >> 63)) as u64;
            assert_eq!(unzigzag(folded), n, "n={n}");
        }
    }

    #[test]
    fn decode_example_1_from_spec() {
        // §D.1: the complete Example 1 file bytes (from the hex dump). Two channels,
        // one interchannel sample; values 25588 and 10416 (§D.1.4). This exercises the
        // whole path: magic, STREAMINFO, frame header CRC-8, two VERBATIM subframes
        // with wasted bits, and the frame CRC-16.
        let file: [u8; 0x39] = [
            0x66, 0x4c, 0x61, 0x43, 0x80, 0x00, 0x00, 0x22, 0x10, 0x00, 0x10, 0x00, 0x00, 0x00,
            0x0f, 0x00, 0x00, 0x0f, 0x0a, 0xc4, 0x42, 0xf0, 0x00, 0x00, 0x00, 0x01, 0x3e, 0x84,
            0xb4, 0x18, 0x07, 0xdc, 0x69, 0x03, 0x07, 0x58, 0x6a, 0x3d, 0xad, 0x1a, 0x2e, 0x0f,
            0xff, 0xf8, 0x69, 0x18, 0x00, 0x00, 0xbf, 0x03, 0x58, 0xfd, 0x03, 0x12, 0x8b, 0xaa,
            0x9a,
        ];
        let dec = FlacDecoder::decode(&file).expect("decode example 1");
        assert_eq!(dec.info.channels, 2);
        assert_eq!(dec.info.bits_per_sample, 16);
        assert_eq!(dec.info.sample_rate, 44100);
        assert_eq!(dec.info.total_samples, 1);
        assert_eq!(dec.samples, vec![25588, 10416]);
    }
}
