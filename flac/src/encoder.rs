//! The FLAC encoder core (spec: RFC 9639 §8, §9; `spec/rfc9639.txt`).
//!
//! [`FlacEncoder`] turns raw **interleaved** integer PCM into a FLAC bitstream:
//! `fLaC` marker, a STREAMINFO metadata block, then one frame per block of samples,
//! each frame carrying one subframe per channel (§6 format layout).
//!
//! The design is deliberately closed-form and allocation-light on the hot path — a
//! frame is built into a reused [`BitWriter`], planar channel scratch is reused
//! across frames, and the per-subframe residual search reuses buffers too. Channels
//! are coded independently (no stereo decorrelation yet); each subframe picks the
//! cheapest of `CONSTANT` / `VERBATIM` / `FIXED` (orders 0–4) by estimated bit cost
//! (§9.2). LPC (§9.2.6) is a future compression win, not required for validity.
//!
//! ## Streaming shape
//! Because STREAMINFO carries stream-wide fields that are only known once the whole
//! input has been seen (min/max block size, min/max frame size, total samples), the
//! encoder emits a *placeholder* STREAMINFO in [`FlacEncoder::new`] and, on
//! [`finish`](FlacEncoder::finish), returns the finalised 34-byte STREAMINFO body so
//! the caller can patch the header in place (its byte offset is fixed and known:
//! [`STREAMINFO_BODY_OFFSET`]). This keeps the encoder single-pass and streaming: it
//! never buffers the whole file.

use crate::bitstream::{crc16, crc8, BitWriter};

/// Interleaved integer PCM sample format the encoder accepts. All are signed; the
/// bit depth maps straight to FLAC's per-sample bit depth (§9.1.4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleFormat {
    S8,
    S16,
    S24,
    S32,
}

impl SampleFormat {
    pub fn bits_per_sample(self) -> u32 {
        match self {
            SampleFormat::S8 => 8,
            SampleFormat::S16 => 16,
            SampleFormat::S24 => 24,
            SampleFormat::S32 => 32,
        }
    }

    /// Bytes per sample in the interleaved little-endian input (S24 is packed 3-byte).
    pub fn bytes_per_sample(self) -> usize {
        match self {
            SampleFormat::S8 => 1,
            SampleFormat::S16 => 2,
            SampleFormat::S24 => 3,
            SampleFormat::S32 => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    /// A constructor argument was outside what FLAC / this encoder supports.
    UnsupportedParams(&'static str),
    /// Input byte length was not a whole number of interchannel samples.
    RaggedInput,
}

/// Byte offset of the STREAMINFO block *body* (the 34 bytes after its 4-byte metadata
/// header) from the start of the stream: 4 (`fLaC`) + 4 (metadata block header).
/// The caller patches the finalised body here after [`FlacEncoder::finish`].
pub const STREAMINFO_BODY_OFFSET: usize = 8;
/// Length of the STREAMINFO body in bytes (§8.2 Table 3): fixed at 34.
pub const STREAMINFO_BODY_LEN: usize = 34;

/// Streamable-subset ceiling on block size for rate <= 48 kHz (§7). We stay within
/// the subset so output is maximally interoperable, and it keeps the block-size bits
/// in the compact coded form.
const MAX_BLOCK_SIZE: u32 = 4608;
/// Largest partition order we search (§7 caps the streamable subset at 8).
const MAX_PARTITION_ORDER: u32 = 8;

pub struct FlacEncoder {
    sample_rate: u32,
    channels: u32,
    format: SampleFormat,
    bits_per_sample: u32,

    // STREAMINFO fields accumulated across the stream (§8.2).
    min_block_size: u32,
    max_block_size: u32,
    min_frame_size: u32,
    max_frame_size: u32,
    total_samples: u64,
    frame_number: u64,
    finished: bool,
    /// Forward-only writer: STREAMINFO was emitted up front with "unknown" (0) frame
    /// sizes and total samples, so per-frame accounting must not revise it.
    streaming: bool,

    // Reused scratch, so steady-state encoding does not allocate per frame.
    planar: Vec<Vec<i64>>,
    frame_writer: BitWriter,
    residual: Vec<i64>,
    part_scratch: Vec<u64>,
    /// Reused winning-residual copy for the subframe FIXED search (was a per-subframe
    /// `Vec::new`), so steady-state encoding does not allocate per frame.
    best_residual: Vec<i64>,
    /// Reused Rice-parameter buffer for the winning residual plan (was `ResidualPlan`'s
    /// owned `params`), so steady-state encoding does not allocate per frame.
    plan_params: Vec<u32>,
    /// Reused per-partition-order candidate Rice params inside the search (was a
    /// per-search `Vec::with_capacity`), cleared and refilled per partition order.
    cand_params: Vec<u32>,
}

impl FlacEncoder {
    /// Create an encoder for interleaved `format` PCM at `sample_rate` Hz with
    /// `channels` channels. Emits the `fLaC` marker and a placeholder STREAMINFO into
    /// the returned header bytes (patch it after [`finish`](Self::finish)).
    ///
    /// Supports 1–8 channels and 8/16/24/32-bit samples; the milestone target is S16
    /// mono+stereo at 44100/48000, but nothing here is limited to that.
    // One-time construction: the per-frame scratch (planar/residual/writer/params) is
    // built once here and reused across frames; the header buffer is emitted once.
    #[allow(clippy::disallowed_methods)]
    pub fn new(
        sample_rate: u32,
        channels: u32,
        format: SampleFormat,
    ) -> Result<(Self, Vec<u8>), EncodeError> {
        if channels < 1 || channels > 8 {
            return Err(EncodeError::UnsupportedParams("channels must be 1..=8"));
        }
        if sample_rate == 0 || sample_rate >= (1 << 20) {
            return Err(EncodeError::UnsupportedParams("sample rate out of u20 range"));
        }
        let bits_per_sample = format.bits_per_sample();

        let enc = Self {
            sample_rate,
            channels,
            format,
            bits_per_sample,
            min_block_size: u32::MAX,
            max_block_size: 0,
            min_frame_size: u32::MAX,
            max_frame_size: 0,
            total_samples: 0,
            frame_number: 0,
            finished: false,
            streaming: false,
            planar: (0..channels).map(|_| Vec::new()).collect(),
            frame_writer: BitWriter::with_capacity(16 * 1024),
            residual: Vec::new(),
            part_scratch: Vec::new(),
            best_residual: Vec::new(),
            plan_params: Vec::new(),
            cand_params: Vec::new(),
        };

        let mut header = Vec::with_capacity(4 + 4 + STREAMINFO_BODY_LEN);
        header.extend_from_slice(b"fLaC");
        // Metadata block header (§8.1): last-block flag = 1 (STREAMINFO is our only
        // metadata block), type 0 (Streaminfo), length 34.
        header.push(0x80); // 0b1_0000000
        header.extend_from_slice(&[0x00, 0x00, STREAMINFO_BODY_LEN as u8]);
        // Placeholder body; patched by the caller with `finish()`'s output.
        header.extend_from_slice(&enc.streaminfo_body());
        debug_assert_eq!(header.len(), STREAMINFO_BODY_OFFSET + STREAMINFO_BODY_LEN);

        Ok((enc, header))
    }

    /// Emit the `fLaC` marker + STREAMINFO for a **streaming**, forward-only writer
    /// (e.g. the [`FlacEnc`](crate::FlacEnc) element writing to a sequential sink that
    /// cannot be back-patched). The advertised max block size is `declared_block`, and
    /// frame sizes / total samples are written as 0 == "unknown", which is spec-legal
    /// (§8.2: "A value of 0 signifies that the value is not known"). The result is a
    /// valid, streamable FLAC header requiring no finalisation.
    // One-time construction: builds the header buffer once, per stream (not per frame).
    #[allow(clippy::disallowed_methods)]
    pub fn new_streaming(
        sample_rate: u32,
        channels: u32,
        format: SampleFormat,
        declared_block: u32,
    ) -> Result<(Self, Vec<u8>), EncodeError> {
        let (mut enc, _placeholder) = Self::new(sample_rate, channels, format)?;
        let block = declared_block.clamp(16, MAX_BLOCK_SIZE);
        // Advertise a fixed block size and "unknown" for everything only known at EOF.
        enc.min_block_size = block;
        enc.max_block_size = block;
        enc.min_frame_size = 0;
        enc.max_frame_size = 0;
        enc.total_samples = 0;
        // Mark this instance so per-frame accounting does not shrink the advertised
        // fields below (streaming header must not be revised after the fact).
        enc.streaming = true;

        let mut header = Vec::with_capacity(4 + 4 + STREAMINFO_BODY_LEN);
        header.extend_from_slice(b"fLaC");
        header.push(0x80);
        header.extend_from_slice(&[0x00, 0x00, STREAMINFO_BODY_LEN as u8]);
        header.extend_from_slice(&enc.streaminfo_body());
        Ok((enc, header))
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn format(&self) -> SampleFormat {
        self.format
    }

    /// Largest block (interchannel samples) the caller should submit per
    /// [`encode_interleaved`](Self::encode_interleaved) call to stay in-subset.
    pub fn max_block_size(&self) -> u32 {
        MAX_BLOCK_SIZE
    }

    /// Encode a run of interleaved little-endian PCM bytes, appending one or more
    /// frames to `out`. The byte length MUST be a whole number of interchannel
    /// samples. Blocks larger than [`max_block_size`](Self::max_block_size) are split
    /// into multiple frames automatically.
    pub fn encode_interleaved(&mut self, bytes: &[u8], out: &mut Vec<u8>) -> Result<(), EncodeError> {
        let bps = self.format.bytes_per_sample();
        let frame_stride = bps * self.channels as usize;
        if frame_stride == 0 || bytes.len() % frame_stride != 0 {
            return Err(EncodeError::RaggedInput);
        }
        let total_frames = bytes.len() / frame_stride; // interchannel samples

        let mut off = 0usize;
        while off < total_frames {
            let block = (total_frames - off).min(MAX_BLOCK_SIZE as usize);
            self.deinterleave(bytes, off, block);
            self.encode_one_block(block, out);
            off += block;
        }
        Ok(())
    }

    /// Finalise the stream. Returns the completed 34-byte STREAMINFO **body** (§8.2)
    /// with the accumulated min/max sizes and total sample count filled in; the
    /// caller writes it over the placeholder at [`STREAMINFO_BODY_OFFSET`].
    ///
    /// FLAC needs no end-of-stream marker (§9: frames just stop), so there is nothing
    /// to append to the frame data.
    pub fn finish(&mut self) -> [u8; STREAMINFO_BODY_LEN] {
        self.finished = true;
        // If no audio was ever submitted, present a coherent empty-stream header.
        if self.max_block_size == 0 {
            self.min_block_size = 0;
            self.max_block_size = 0;
            self.min_frame_size = 0;
            self.max_frame_size = 0;
        }
        self.streaminfo_body()
    }

    // --- STREAMINFO ---------------------------------------------------------

    /// Serialise the current STREAMINFO body (§8.2 Table 3). MD5 is left all-zeros
    /// ("unknown"), which is spec-legal (§8.2: "A value of 0 signifies ... not known").
    fn streaminfo_body(&self) -> [u8; STREAMINFO_BODY_LEN] {
        let mut w = BitWriter::with_capacity(STREAMINFO_BODY_LEN);
        let min_block = if self.min_block_size == u32::MAX { 0 } else { self.min_block_size };
        let min_frame = if self.min_frame_size == u32::MAX { 0 } else { self.min_frame_size };
        w.write_bits(min_block as u64, 16); // min block size
        w.write_bits(self.max_block_size as u64, 16); // max block size
        w.write_bits(min_frame as u64, 24); // min frame size (0 = unknown)
        w.write_bits(self.max_frame_size as u64, 24); // max frame size (0 = unknown)
        w.write_bits(self.sample_rate as u64, 20); // sample rate
        w.write_bits((self.channels - 1) as u64, 3); // channels - 1
        w.write_bits((self.bits_per_sample - 1) as u64, 5); // bits per sample - 1
        w.write_bits(self.total_samples, 36); // total interchannel samples (0 = unknown)
        w.write_bits(0, 64); // MD5 high 64 bits (unknown)
        w.write_bits(0, 64); // MD5 low 64 bits (unknown)
        let bytes = w.into_bytes();
        debug_assert_eq!(bytes.len(), STREAMINFO_BODY_LEN);
        let mut body = [0u8; STREAMINFO_BODY_LEN];
        body.copy_from_slice(&bytes);
        body
    }

    // --- Deinterleaving -----------------------------------------------------

    /// Extract `block` interchannel samples starting at interchannel index `start`
    /// into the planar per-channel scratch buffers, decoding little-endian samples of
    /// the configured width into sign-extended i64.
    fn deinterleave(&mut self, bytes: &[u8], start: usize, block: usize) {
        let bps = self.format.bytes_per_sample();
        let ch = self.channels as usize;
        let stride = bps * ch;
        for c in 0..ch {
            let dst = &mut self.planar[c];
            dst.clear();
            dst.reserve(block);
        }
        for i in 0..block {
            let base = (start + i) * stride;
            for c in 0..ch {
                let s = base + c * bps;
                let v = read_sample_le(&bytes[s..s + bps], self.format);
                self.planar[c].push(v);
            }
        }
    }

    // --- Frame encoding -----------------------------------------------------

    fn encode_one_block(&mut self, block: usize, out: &mut Vec<u8>) {
        let block_u32 = block as u32;
        if !self.streaming {
            self.min_block_size = self.min_block_size.min(block_u32);
            self.max_block_size = self.max_block_size.max(block_u32);
        }

        let w = &mut self.frame_writer;
        // Reset the reused writer by moving out its buffer; cheaper than allocating a
        // fresh one, and keeps capacity.
        *w = BitWriter::with_capacity(0);

        self.write_frame_header(block_u32);
        // Encode each channel's subframe into the same bit-packed frame (§9.2:
        // subframes appear serially and need not be byte aligned).
        for c in 0..self.channels as usize {
            // Take the planar buffer out to satisfy the borrow checker, then restore.
            let samples = std::mem::take(&mut self.planar[c]);
            Self::write_subframe(
                &mut self.frame_writer,
                &samples,
                self.bits_per_sample,
                &mut self.residual,
                &mut self.part_scratch,
                &mut self.best_residual,
                &mut self.plan_params,
                &mut self.cand_params,
            );
            self.planar[c] = samples;
        }
        // Frame footer (§9.3): pad to a byte boundary, then CRC-16 over the whole frame.
        self.frame_writer.align();
        let frame_bytes = self.frame_writer.as_aligned_bytes();
        let crc = crc16(frame_bytes);
        let frame_len = frame_bytes.len() + 2;

        out.extend_from_slice(frame_bytes);
        out.extend_from_slice(&crc.to_be_bytes());

        if !self.streaming {
            let flen = frame_len as u32;
            self.min_frame_size = self.min_frame_size.min(flen);
            self.max_frame_size = self.max_frame_size.max(flen);
            self.total_samples += block as u64;
        }
        self.frame_number += 1;
    }

    /// Frame header (§9.1). Fixed block size stream (blocking strategy 0), so the
    /// coded number is a frame number (§9.1.5). Uses the 8-bit "uncommon" block-size
    /// escape for arbitrary block sizes and a coded sample-rate value where possible.
    fn write_frame_header(&mut self, block: u32) {
        let w = &mut self.frame_writer;
        // Sync code 0b111111111111100 (15 bits) + blocking strategy bit 0 (§9.1).
        // Together the first two bytes are 0xFFF8 for a fixed block size stream.
        w.write_bits(0b111111111111100, 15);
        w.write_bits(0, 1); // blocking strategy: fixed block size

        // Block size bits (§9.1.1) and sample rate bits (§9.1.2).
        let (bs_bits, bs_extra) = block_size_code(block);
        let (sr_bits, sr_extra) = sample_rate_code(self.sample_rate);
        w.write_bits(bs_bits as u64, 4);
        w.write_bits(sr_bits as u64, 4);

        // Channel bits (§9.1.3): independent channels, so value = channels - 1 for the
        // "N channels: ..." rows (0b0000..0b0111). No stereo decorrelation.
        w.write_bits((self.channels - 1) as u64, 4);

        // Bit depth bits (§9.1.4) + one mandatory reserved 0 bit.
        w.write_bits(bit_depth_code(self.bits_per_sample) as u64, 3);
        w.write_bits(0, 1); // reserved, MUST be zero

        // Coded number = frame number (§9.1.5), UTF-8-like variable length.
        write_coded_number(w, self.frame_number);

        // Uncommon block size / sample rate extensions, in that order (§9.1.6, §9.1.7).
        if let Some((val, bits)) = bs_extra {
            w.write_bits(val as u64, bits);
        }
        if let Some((val, bits)) = sr_extra {
            w.write_bits(val as u64, bits);
        }

        // Frame header CRC-8 over everything written so far (§9.1.8). The header is
        // byte-aligned at this point (all fields above are whole bytes for our codes).
        debug_assert!(w.is_byte_aligned());
        let crc = crc8(w.as_aligned_bytes());
        w.write_bits(crc as u64, 8);
    }

    /// Encode one channel as the cheapest available subframe type (§9.2). `samples`
    /// are already sign-extended to i64; `bps` is the subframe bit depth.
    #[allow(clippy::too_many_arguments)] // all buffers are reused scratch threaded to avoid per-frame allocation
    fn write_subframe(
        w: &mut BitWriter,
        samples: &[i64],
        bps: u32,
        residual: &mut Vec<i64>,
        part_scratch: &mut Vec<u64>,
        best_residual: &mut Vec<i64>,
        plan_params: &mut Vec<u32>,
        cand_params: &mut Vec<u32>,
    ) {
        let n = samples.len();

        // CONSTANT: all samples identical (§9.2.3). Cheapest when applicable.
        if n > 0 && samples.iter().all(|&s| s == samples[0]) {
            // Subframe header: 0 bit + type 0b000000 + wasted-bits flag 0.
            w.write_bits(0, 1);
            w.write_bits(0b000000, 6);
            w.write_bits(0, 1);
            w.write_signed(samples[0], bps);
            return;
        }

        // Evaluate FIXED predictors of order 0..=4 and pick the one with the smallest
        // estimated residual bit cost (§9.2.5). Fall back to VERBATIM if even the best
        // FIXED order does not beat storing samples raw, or if any residual would
        // exceed the 32-bit limit (§9.2.7.3).
        let max_order = 4.min(n.saturating_sub(1));
        let mut best_order: Option<u32> = None;
        let mut best_bits = u64::MAX;
        best_residual.clear();
        plan_params.clear();
        let mut best_res_cost = ResidualPlan::default();

        for order in 0..=max_order as u32 {
            compute_fixed_residual(samples, order, residual);
            if !residual_in_range(residual) {
                continue;
            }
            // `best_residual_cost` writes the winning partition's Rice params into
            // `cand_params` (reused scratch); we snapshot them into `plan_params` only
            // when this order wins overall — no per-order/per-subframe allocation.
            let (res_bits, plan) = best_residual_cost(residual, n, order, part_scratch, cand_params);
            // Subframe cost = header (7 bits + warm-up samples) + residual bits.
            let warmup_bits = order as u64 * bps as u64;
            let total = 8 + warmup_bits + res_bits;
            if total < best_bits {
                best_bits = total;
                best_order = Some(order);
                best_residual.clear();
                best_residual.extend_from_slice(residual);
                plan_params.clear();
                plan_params.extend_from_slice(cand_params);
                best_res_cost = plan;
            }
        }

        // VERBATIM cost (§9.2.4): header (8 bits) + n * bps.
        let verbatim_bits = 8 + n as u64 * bps as u64;

        match best_order {
            Some(order) if best_bits <= verbatim_bits => {
                Self::write_fixed_subframe(w, samples, order, bps, best_residual, plan_params, &best_res_cost);
            }
            _ => Self::write_verbatim_subframe(w, samples, bps),
        }
    }

    fn write_verbatim_subframe(w: &mut BitWriter, samples: &[i64], bps: u32) {
        // Header: 0 bit + type 0b000001 + wasted-bits flag 0 (§9.2.1, §9.2.4).
        w.write_bits(0, 1);
        w.write_bits(0b000001, 6);
        w.write_bits(0, 1);
        for &s in samples {
            w.write_signed(s, bps);
        }
    }

    fn write_fixed_subframe(
        w: &mut BitWriter,
        samples: &[i64],
        order: u32,
        bps: u32,
        residual: &[i64],
        params: &[u32],
        plan: &ResidualPlan,
    ) {
        // Header: 0 bit + type 0b001ooo (v = 8 + order) + wasted-bits flag 0 (§9.2.1).
        w.write_bits(0, 1);
        w.write_bits((0b001000 | order) as u64, 6);
        w.write_bits(0, 1);
        // Warm-up samples, unencoded (§9.2.5 Table 21).
        for &s in &samples[..order as usize] {
            w.write_signed(s, bps);
        }
        write_partitioned_residual(w, residual, samples.len(), order, params, plan);
    }
}

/// A chosen residual-coding plan: partition order plus the Rice-parameter field width
/// (§9.2.7). Built by [`best_residual_cost`] and replayed by
/// [`write_partitioned_residual`], so the search cost is not paid twice. The per-
/// partition Rice parameters themselves live in a caller-owned reusable buffer (the
/// encoder's `plan_params`), passed alongside this plan — never a per-subframe `Vec`.
#[derive(Default, Clone, Copy)]
struct ResidualPlan {
    partition_order: u32,
    /// 4- or 5-bit Rice parameters (§9.2.7 Table 23). 4 unless a param needs 5 bits.
    param_bits: u32,
}

/// Read one little-endian signed sample of the given width, sign-extended to i64.
fn read_sample_le(bytes: &[u8], format: SampleFormat) -> i64 {
    match format {
        SampleFormat::S8 => bytes[0] as i8 as i64,
        SampleFormat::S16 => i16::from_le_bytes([bytes[0], bytes[1]]) as i64,
        SampleFormat::S24 => {
            let raw = (bytes[0] as u32) | ((bytes[1] as u32) << 8) | ((bytes[2] as u32) << 16);
            // Sign-extend from bit 23.
            if raw & 0x0080_0000 != 0 {
                (raw | 0xFF00_0000) as i32 as i64
            } else {
                raw as i64
            }
        }
        SampleFormat::S32 => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64,
    }
}

/// FIXED-predictor residual for a given order (§9.2.5 Table 20). The first `order`
/// entries of `out` are the warm-up positions and are left as the raw samples (they
/// are stored unencoded, not via the residual coder); residual proper starts at
/// index `order`. `out` is sized to `samples.len()` with warm-up slots included so
/// callers can index uniformly.
///
/// Uses i64 arithmetic throughout, which cannot overflow for our <=32-bit inputs
/// (Appendix A: order-4 needs at most bps+4 bits), so the §9.2.7.3 range check below
/// is about the *32-bit residual limit*, never about i64 overflow.
fn compute_fixed_residual(samples: &[i64], order: u32, out: &mut Vec<i64>) {
    let n = samples.len();
    out.clear();
    out.resize(n, 0);
    let o = order as usize;
    for i in 0..o.min(n) {
        out[i] = samples[i]; // warm-up placeholder (not emitted through the coder)
    }
    match order {
        0 => {
            for i in 0..n {
                out[i] = samples[i];
            }
        }
        1 => {
            for i in o..n {
                out[i] = samples[i] - samples[i - 1];
            }
        }
        2 => {
            for i in o..n {
                out[i] = samples[i] - 2 * samples[i - 1] + samples[i - 2];
            }
        }
        3 => {
            for i in o..n {
                out[i] = samples[i] - 3 * samples[i - 1] + 3 * samples[i - 2] - samples[i - 3];
            }
        }
        4 => {
            for i in o..n {
                out[i] = samples[i] - 4 * samples[i - 1] + 6 * samples[i - 2] - 4 * samples[i - 3]
                    + samples[i - 4];
            }
        }
        _ => unreachable!("fixed order 0..=4"),
    }
}

/// §9.2.7.3: every residual sample must fit a 32-bit two's-complement integer
/// excluding the most negative value (|r| < 2^31). We only check the residual proper;
/// warm-up placeholders are stored as full-width samples, not residuals.
fn residual_in_range(residual: &[i64]) -> bool {
    const LIMIT: i64 = 1 << 31;
    residual.iter().all(|&r| r > -LIMIT && r < LIMIT)
}

/// Zigzag ("folded") mapping of a signed residual to an unsigned Rice symbol
/// (§9.2.7.2): non-negative n -> 2n, negative n -> -2n-1.
#[inline]
fn zigzag(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Search partition orders and Rice parameters for the residual of a subframe, and
/// return the total residual bit cost together with the winning [`ResidualPlan`]
/// (§9.2.7). `n` is the block size; `order` the predictor order (so the first
/// partition holds `(n >> po) - order` samples). The winning partition's Rice
/// parameters are written into `out_params` (a caller-owned reusable buffer, cleared
/// first) so nothing is allocated per subframe.
fn best_residual_cost(
    residual: &[i64],
    n: usize,
    order: u32,
    scratch: &mut Vec<u64>,
    out_params: &mut Vec<u32>,
) -> (u64, ResidualPlan) {
    // Precompute folded residuals once (indices `order..n`).
    scratch.clear();
    scratch.reserve(n - order as usize);
    for &r in &residual[order as usize..n] {
        scratch.push(zigzag(r));
    }

    let mut best_bits = u64::MAX;
    let mut best_plan = ResidualPlan::default();
    out_params.clear();

    // Only partition orders where n is divisible by 2^po and each partition is larger
    // than the predictor order are valid (§9.2.7). Search 0..=MAX_PARTITION_ORDER.
    for po in 0..=MAX_PARTITION_ORDER {
        let parts = 1u32 << po;
        if n % parts as usize != 0 {
            continue;
        }
        let part_len = n / parts as usize;
        if part_len <= order as usize {
            break; // (block >> po) must exceed predictor order; larger po only shrinks it
        }

        let mut max_param = 0u32;
        // Partition parameter overhead is added after we know param_bits.
        let mut coded_bits = 0u64;
        let mut folded_idx = 0usize;
        for p in 0..parts as usize {
            // First partition is short by `order` (warm-up not coded); others full.
            let count = if p == 0 { part_len - order as usize } else { part_len };
            let slice = &scratch[folded_idx..folded_idx + count];
            folded_idx += count;
            let (param, bits) = best_rice_param(slice);
            coded_bits += bits;
            max_param = max_param.max(param);
        }

        // Rice parameter field is 4 bits, or 5 if any parameter needs 5 bits (§9.2.7).
        let param_bits = if max_param <= 14 { 4 } else { 5 };
        // 2 bits coding method + 4 bits partition order + one parameter field per
        // partition + the coded residual bits.
        let total = 2 + 4 + (parts as u64) * param_bits as u64 + coded_bits;
        if total < best_bits {
            best_bits = total;
            best_plan = ResidualPlan { partition_order: po, param_bits };
            // Refill `out_params` for the new winner (cheap re-run of the convex Rice
            // search; only happens on an improving partition order, a few times total).
            out_params.clear();
            let mut folded_idx = 0usize;
            for p in 0..parts as usize {
                let count = if p == 0 { part_len - order as usize } else { part_len };
                let slice = &scratch[folded_idx..folded_idx + count];
                folded_idx += count;
                let (param, _) = best_rice_param(slice);
                out_params.push(param);
            }
        }
    }

    (best_bits, best_plan)
}

/// Best Rice parameter for a partition of already-folded residuals, and the total
/// bits that parameter costs (unary quotients + fixed remainders) (§9.2.7.2).
///
/// Cost(k) = count * (k + 1) + sum(folded >> k) and is convex in k, minimised near
/// where `count << k ≈ sum` (i.e. the mean folded value sits around 2^k). We seed k
/// from that estimate and scan the small convex neighbourhood, evaluating the exact
/// bit count on each candidate — no full 0..=30 sweep, no floating point.
fn best_rice_param(folded: &[u64]) -> (u32, u64) {
    if folded.is_empty() {
        return (0, 0);
    }
    let count = folded.len() as u64;
    let sum: u64 = folded.iter().copied().sum();

    // Seed: largest k with (count << k) <= sum, i.e. floor(log2(sum / count)).
    let mut seed = 0u32;
    while seed < 30 && (count << (seed + 1)) <= sum {
        seed += 1;
    }
    // Exact cost for a given k (bounded k keeps remainders within 32 bits and avoids
    // the 0b1111 escape).
    let cost = |k: u32| -> u64 {
        let mut bits = count * (k as u64 + 1);
        for &f in folded {
            bits += f >> k;
        }
        bits
    };

    // Walk down then up from the seed while it improves — convexity bounds this to a
    // couple of steps in practice.
    let mut best_k = seed;
    let mut best = cost(seed);
    let mut k = seed;
    while k > 0 {
        let c = cost(k - 1);
        if c < best {
            best = c;
            best_k = k - 1;
            k -= 1;
        } else {
            break;
        }
    }
    let mut k = seed;
    while k < 30 {
        let c = cost(k + 1);
        if c < best {
            best = c;
            best_k = k + 1;
            k += 1;
        } else {
            break;
        }
    }
    (best_k, best)
}

/// Emit the partitioned Rice-coded residual for a subframe using a plan produced by
/// [`best_residual_cost`] (§9.2.7). Method is always partitioned Rice (never escape,
/// since a finite Rice cost is guaranteed by the range check).
fn write_partitioned_residual(
    w: &mut BitWriter,
    residual: &[i64],
    n: usize,
    order: u32,
    params: &[u32],
    plan: &ResidualPlan,
) {
    // Coding method (§9.2.7 Table 23): 0b00 for 4-bit params, 0b01 for 5-bit.
    let method = if plan.param_bits == 4 { 0b00 } else { 0b01 };
    w.write_bits(method, 2);
    w.write_bits(plan.partition_order as u64, 4);

    let parts = 1usize << plan.partition_order;
    let part_len = n / parts;
    let mut idx = order as usize; // residual proper starts after warm-up
    for p in 0..parts {
        let count = if p == 0 { part_len - order as usize } else { part_len };
        let param = params[p];
        w.write_bits(param as u64, plan.param_bits);
        for _ in 0..count {
            let folded = zigzag(residual[idx]);
            idx += 1;
            let q = (folded >> param) as u32;
            w.write_unary(q); // most-significant part, unary
            if param > 0 {
                w.write_bits(folded & ((1u64 << param) - 1), param); // remainder
            }
        }
    }
}

// --- Frame header field encoders (§9.1) ------------------------------------

/// Block size bits (§9.1.1 Table 14). Returns the 4-bit code and any extra field
/// `(value, bit_width)` that follows the coded number. Common table values use the
/// compact code; everything else uses the 8- or 16-bit "uncommon" escape.
fn block_size_code(block: u32) -> (u8, Option<(u32, u32)>) {
    match block {
        192 => (0b0001, None),
        576 => (0b0010, None),
        1152 => (0b0011, None),
        2304 => (0b0100, None),
        4608 => (0b0101, None),
        256 => (0b1000, None),
        512 => (0b1001, None),
        1024 => (0b1010, None),
        2048 => (0b1011, None),
        4096 => (0b1100, None),
        8192 => (0b1101, None),
        16384 => (0b1110, None),
        32768 => (0b1111, None),
        // Uncommon block size, stored as (block - 1). Prefer the 8-bit escape when it
        // fits (§9.1.6), else the 16-bit escape.
        _ if block >= 1 && block <= 256 => (0b0110, Some((block - 1, 8))),
        _ => (0b0111, Some((block - 1, 16))),
    }
}

/// Sample rate bits (§9.1.2 Table 15). Returns the 4-bit code and any extra field.
/// Common rates use the compact code; others use the 8-bit-kHz / 16-bit-Hz escapes.
fn sample_rate_code(rate: u32) -> (u8, Option<(u32, u32)>) {
    match rate {
        88200 => (0b0001, None),
        176400 => (0b0010, None),
        192000 => (0b0011, None),
        8000 => (0b0100, None),
        16000 => (0b0101, None),
        22050 => (0b0110, None),
        24000 => (0b0111, None),
        32000 => (0b1000, None),
        44100 => (0b1001, None),
        48000 => (0b1010, None),
        96000 => (0b1011, None),
        // Uncommon rate escapes: whole kHz in 8 bits, else Hz in 16 bits, else Hz/10
        // in 16 bits (§9.1.7). We never use streaminfo-relative (0b0000) so output
        // stays streamable (§7).
        _ if rate % 1000 == 0 && rate / 1000 <= 255 => (0b1100, Some((rate / 1000, 8))),
        _ if rate <= 0xFFFF => (0b1101, Some((rate, 16))),
        _ => (0b1110, Some((rate / 10, 16))),
    }
}

/// Bit depth bits (§9.1.4 Table 17). Falls back to 0b000 (streaminfo) only for depths
/// with no code, which our supported formats never hit.
fn bit_depth_code(bits: u32) -> u8 {
    match bits {
        8 => 0b001,
        12 => 0b010,
        16 => 0b100,
        20 => 0b101,
        24 => 0b110,
        32 => 0b111,
        _ => 0b000,
    }
}

/// Write a coded number in the UTF-8-like extended encoding (§9.1.5 Table 18). Used
/// for the frame number (fixed block size) — must fit 31 bits unencoded.
fn write_coded_number(w: &mut BitWriter, value: u64) {
    if value < 0x80 {
        w.write_bits(value, 8);
        return;
    }
    // Determine how many continuation bytes are needed and the lead-byte prefix.
    // Ranges from Table 18.
    let (lead_bits, total_bytes) = if value < 0x800 {
        (5, 2) // 110xxxxx
    } else if value < 0x1_0000 {
        (4, 3) // 1110xxxx
    } else if value < 0x20_0000 {
        (3, 4) // 11110xxx
    } else if value < 0x400_0000 {
        (2, 5) // 111110xx
    } else if value < 0x8000_0000 {
        (1, 6) // 1111110x
    } else {
        (0, 7) // 11111110, then 6 continuation bytes (up to 36 bits)
    };
    let cont = total_bytes - 1;
    // Lead byte (§9.1.5 Table 18): `total_bytes` leading one bits, a separating zero
    // bit, then the top `lead_bits` of the value. (A 2-byte code leads with 0b110, a
    // 7-byte code with 0b11111110 — i.e. one 1-bit per byte in the sequence.)
    let mut lead: u64 = 0;
    for _ in 0..total_bytes {
        lead = (lead << 1) | 1;
    }
    lead <<= 1; // the separating 0 bit
    let value_bits = (cont * 6) as u32; // bits carried by continuation bytes
    if lead_bits > 0 {
        let top = value >> value_bits;
        lead = (lead << lead_bits) | (top & ((1u64 << lead_bits) - 1));
    }
    w.write_bits(lead, 8);
    // Continuation bytes: 10xxxxxx, six value bits each, most significant first.
    for i in (0..cont).rev() {
        let shift = (i * 6) as u32;
        let six = (value >> shift) & 0x3F;
        w.write_bits(0b10_000000 | six, 8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitstream::BitReader;

    #[test]
    fn zigzag_matches_spec_examples() {
        // §9.2.7.2: positive n -> 2n; negative n -> -2n-1.
        assert_eq!(zigzag(0), 0);
        assert_eq!(zigzag(1), 2);
        assert_eq!(zigzag(-1), 1);
        assert_eq!(zigzag(2), 4);
        assert_eq!(zigzag(-2), 3);
        assert_eq!(zigzag(19), 38); // the §9.2.7.2 worked value (folded = 38)
    }

    #[test]
    fn coded_number_single_byte() {
        for v in [0u64, 1, 63, 127] {
            let mut w = BitWriter::new();
            write_coded_number(&mut w, v);
            let bytes = w.into_bytes();
            assert_eq!(bytes, vec![v as u8], "value {v}");
        }
    }

    #[test]
    fn coded_number_51_billion_matches_spec() {
        // §9.1.5 worked example: 51 billion samples -> the 7-byte sequence shown.
        let mut w = BitWriter::new();
        write_coded_number(&mut w, 51_000_000_000);
        let bytes = w.into_bytes();
        assert_eq!(
            bytes,
            vec![0b11111110, 0b10101111, 0b10011111, 0b10110101, 0b10100011, 0b10111000, 0b10000000],
        );
    }

    #[test]
    fn coded_number_roundtrip_via_reader() {
        // Cross-check a spread of values by decoding them back the way the decoder does.
        for v in [0u64, 1, 127, 128, 200, 2047, 2048, 65535, 65536, 1 << 20, (1u64 << 31) - 1] {
            let mut w = BitWriter::new();
            write_coded_number(&mut w, v);
            let bytes = w.into_bytes();
            let mut r = BitReader::new(&bytes);
            let got = crate::decoder::read_coded_number(&mut r).unwrap();
            assert_eq!(got, v, "value {v}");
        }
    }

    #[test]
    fn streaminfo_body_layout_example_1() {
        // Reproduce §D.1.3 Table 28: stereo, 16-bit, 44100 Hz, block 4096, 1 sample,
        // frame size 15. The MD5 differs (we write zeros), so compare the first 18
        // bytes (everything up to the MD5 field).
        let (mut enc, _hdr) = FlacEncoder::new(44100, 2, SampleFormat::S16).unwrap();
        enc.min_block_size = 4096;
        enc.max_block_size = 4096;
        enc.min_frame_size = 15;
        enc.max_frame_size = 15;
        enc.total_samples = 1;
        let body = enc.streaminfo_body();
        let expected_prefix = [
            0x10, 0x00, // min block 4096
            0x10, 0x00, // max block 4096
            0x00, 0x00, 0x0f, // min frame 15
            0x00, 0x00, 0x0f, // max frame 15
            0x0a, 0xc4, 0x42, // sample rate 44100 (20 bits) | channels-1 (3) | bps-1 hi
            0xf0, // ...continuing bps and start of total samples
            0x00, 0x00, 0x00, 0x01, // total samples 1 (low bits)
        ];
        assert_eq!(&body[..18], &expected_prefix);
        // MD5 (last 16 bytes) is all zeros == "unknown".
        assert_eq!(&body[18..], &[0u8; 16]);
    }

    #[test]
    fn frame_header_crc_matches_example_1() {
        // A stereo 44100 S16 frame with block size 1 (one interchannel sample) and
        // frame number 0 must produce the §D.1.4 header bytes ff f8 69 18 00 00 with
        // CRC-8 0xbf. We drive `write_frame_header` directly.
        let (mut enc, _hdr) = FlacEncoder::new(44100, 2, SampleFormat::S16).unwrap();
        enc.frame_writer = BitWriter::new();
        enc.write_frame_header(1);
        let bytes = enc.frame_writer.into_bytes();
        assert_eq!(bytes, vec![0xff, 0xf8, 0x69, 0x18, 0x00, 0x00, 0xbf]);
    }

    #[test]
    fn read_sample_le_sign_extends() {
        assert_eq!(read_sample_le(&[0xFF], SampleFormat::S8), -1);
        assert_eq!(read_sample_le(&[0x00, 0x80], SampleFormat::S16), -32768);
        assert_eq!(read_sample_le(&[0xFF, 0xFF, 0xFF], SampleFormat::S24), -1);
        assert_eq!(read_sample_le(&[0x00, 0x00, 0x80], SampleFormat::S24), -(1 << 23));
        assert_eq!(
            read_sample_le(&[0xFF, 0xFF, 0xFF, 0xFF], SampleFormat::S32),
            -1
        );
    }
}
