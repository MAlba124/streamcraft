//! `OpusEncoder` — the top-level orchestrator that turns a config + a 48 kHz PCM stream into
//! Opus packets (RFC 6716). It ties the vendored `oxideav-opus` per-mode packet encoders together
//! with mode selection, bandwidth selection, and rate control — the layer libopus calls
//! `opus_encoder.c` (clean-room: built from the RFC, never from that source).
//!
//! # Scope of this milestone (CELT-only)
//!
//! This first cut wires **CELT-only** mode (RFC 6716 §4.3): the MDCT music/low-delay path. It
//! covers mono and stereo, the four CELT frame sizes (2.5/5/10/20 ms), all CELT bandwidths
//! (NB/WB/SWB/FB), and **CBR** rate control. CELT-only is the natural first target for
//! transcode-to-Opus because it takes 48 kHz interleaved `i16` directly (no encode-side
//! resampling — item 6 of the handoff), and because CELT never touches the SILK excitation path,
//! so the shared-`Excitation`/`DecodeBump` gotcha in the handoff is **dormant for this mode** (it
//! only bites SILK-only + Hybrid). SILK-only and Hybrid modes — plus the generic-allocator fix
//! that gotcha needs — are the next milestone; the mode-selection scaffold below is shaped so they
//! slot in without disturbing this path.
//!
//! # Input convention
//!
//! Interleaved **48 kHz `i16`** (the CELT internal rate and the rate `OpusDec` decodes to). An
//! arbitrary-rate source reaches 48 kHz through an upstream `audioresample`, exactly as the decode
//! side leaves 48 kHz → downstream-resample: the [`crate::opusenc::OpusEnc`] element constrains its
//! sink caps to `rate=48000, sample=s16` so autoplug inserts the conversion.
//!
//! # Rate control
//!
//! CBR: the target bitrate and frame duration fix a constant per-packet byte budget
//! (RFC 6716 §2.1.4 rate control; §3.2.5 CBR framing). `payload_bytes = bitrate · frame_s / 8 − 1`
//! (the `−1` is the §3.1 TOC byte) clamped to the §3.2.1 `2..=1275` frame-payload range; CELT's
//! range coder pads to exactly that size (`finish_fixed`), so every packet is `1 + payload_bytes`
//! bytes. VBR/CVBR (the bit reservoir) is a follow-up.

use oxideav_opus::celt_packet_encode::CeltEncoder;
use oxideav_opus::toc::{Bandwidth, Mode};
use oxideav_opus::Error;

/// Opus decode/encode output rate: always 48 kHz (RFC 6716 §2).
pub const OPUS_RATE: u32 = 48_000;
/// §3.2.1 maximum Opus frame payload.
const MAX_FRAME_BYTES: usize = 1275;

/// The application intent, mirroring libopus's three `OPUS_APPLICATION_*` presets. It biases mode
/// and bandwidth selection (RFC 6716 §2 Table 1 gives the bitrate/bandwidth → mode guidance).
///
/// Until SILK/Hybrid are wired, all three resolve to CELT-only; the distinction still steers
/// bandwidth (speech stays narrower) and is the hook the later modes attach to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Application {
    /// Speech / conferencing. Will select SILK once wired; today: CELT-only WB.
    Voip,
    /// General audio / music. CELT-only FB.
    Audio,
    /// Lowest algorithmic delay (no SILK lookahead). CELT-only FB — CELT *is* the low-delay mode.
    LowDelay,
}

/// Encoder configuration. Construct with sensible defaults via [`EncoderConfig::new`] and override
/// fields, or build directly.
#[derive(Debug, Clone, Copy)]
pub struct EncoderConfig {
    /// Input sample rate. This milestone requires **48000** (CELT's native rate); other rates are
    /// rejected — resample upstream.
    pub sample_rate: u32,
    /// Channel count: 1 (mono) or 2 (stereo).
    pub channels: u8,
    /// Target **total** bitrate in bits/s (includes the TOC byte). The CBR budget is derived from
    /// this and the frame duration.
    pub bitrate_bps: u32,
    /// Application intent (biases mode/bandwidth selection).
    pub application: Application,
    /// Frame duration in tenths of a millisecond — one of 25/50/100/200 (2.5/5/10/20 ms) for the
    /// CELT-only path (RFC 6716 §3.1 Table 2, configs 16–31).
    pub frame_ms_tenths: u16,
}

impl EncoderConfig {
    /// A config with common defaults: 48 kHz, [`Application::Audio`], 20 ms frames, and a
    /// bitrate scaled to the channel count (64 kbps mono / 96 kbps stereo — libopus-typical
    /// transparent-ish music rates).
    #[must_use]
    pub fn new(channels: u8) -> Self {
        Self {
            sample_rate: OPUS_RATE,
            channels,
            bitrate_bps: if channels >= 2 { 96_000 } else { 64_000 },
            application: Application::Audio,
            frame_ms_tenths: 200,
        }
    }
}

/// The per-mode encoder actually driven. Only CELT-only is wired this milestone; SILK-only and
/// Hybrid are the placeholders the next milestone fills (see the module docs).
enum ModeImpl {
    Celt(CeltEncoder),
}

/// A unified Opus encoder: PCM frames in, one Opus packet out per frame.
pub struct OpusEncoder {
    channels: usize,
    /// Per-channel samples consumed per [`Self::encode_frame`] call (= 48 kHz frame duration).
    frame_samples: usize,
    /// The constant CBR payload budget (bytes after the TOC), `2..=1275`.
    payload_bytes: usize,
    mode: Mode,
    bandwidth: Bandwidth,
    inner: ModeImpl,
}

impl OpusEncoder {
    /// Build an encoder for `cfg`. Errors ([`Error::MalformedPacket`]) on an unsupported
    /// configuration: a non-48 kHz rate, a channel count outside 1..=2, or a frame duration that
    /// is not a legal CELT frame size.
    pub fn new(cfg: EncoderConfig) -> Result<Self, Error> {
        if cfg.sample_rate != OPUS_RATE {
            // CELT is 48 kHz; the element constrains caps so upstream resamples. A direct caller
            // must feed 48 kHz.
            return Err(Error::MalformedPacket);
        }
        let channels = cfg.channels as usize;
        if !(1..=2).contains(&channels) {
            return Err(Error::MalformedPacket);
        }
        let stereo = channels == 2;

        let (mode, bandwidth) = select_mode(cfg);
        // This milestone only implements CELT-only; `select_mode` never returns another mode yet.
        debug_assert_eq!(mode, Mode::CeltOnly);

        let celt = CeltEncoder::new(bandwidth, cfg.frame_ms_tenths, stereo)?;
        let frame_samples = celt.frame_samples();
        let payload_bytes = cbr_payload_bytes(cfg.bitrate_bps, cfg.frame_ms_tenths);

        Ok(Self {
            channels,
            frame_samples,
            payload_bytes,
            mode,
            bandwidth,
            inner: ModeImpl::Celt(celt),
        })
    }

    /// Per-channel samples one [`Self::encode_frame`] call consumes (48 kHz frame duration).
    #[must_use]
    pub fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    /// Channel count (1 or 2).
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Total interleaved `i16` values one [`Self::encode_frame`] call consumes
    /// (`channels · frame_samples()`).
    #[must_use]
    pub fn frame_len(&self) -> usize {
        self.channels * self.frame_samples
    }

    /// The constant per-packet payload budget in bytes (the packet is `1 + this`).
    #[must_use]
    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    /// The selected operating mode.
    #[must_use]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The selected audio bandwidth.
    #[must_use]
    pub fn bandwidth(&self) -> Bandwidth {
        self.bandwidth
    }

    /// Encode exactly one frame of interleaved 48 kHz `i16` PCM (`frame_len()` values) into one
    /// Opus packet, written into `out` (cleared first). `out` is a caller-owned reusable buffer —
    /// its capacity is retained across calls so a steady encode does no per-frame allocation on
    /// *this* side (the underlying CELT block still allocates internally; making it pool-backed is
    /// the alloc-free follow-up, mirroring the decoder's own trajectory).
    pub fn encode_frame(&mut self, pcm: &[i16], out: &mut Vec<u8>) -> Result<(), Error> {
        if pcm.len() != self.frame_len() {
            return Err(Error::MalformedPacket);
        }
        match &mut self.inner {
            ModeImpl::Celt(celt) => {
                // Writes the whole `1 + payload_bytes` packet straight into `out` (cleared inside),
                // through the encoder's persistent range coder — no per-frame allocation.
                let _info = celt.encode_packet_into(pcm, self.payload_bytes, out)?;
            }
        }
        Ok(())
    }

    /// Reset all carried inter-frame state (stream start / after a seek — RFC 6716 §4.5.2).
    pub fn reset(&mut self) {
        match &mut self.inner {
            ModeImpl::Celt(celt) => celt.reset(),
        }
    }
}

/// Select the operating mode and audio bandwidth for a config (RFC 6716 §2 Table 1). This
/// milestone always returns CELT-only; the bandwidth still tracks the application (speech narrower)
/// so the choice is meaningful and the SILK/Hybrid branches have a clear home.
fn select_mode(cfg: EncoderConfig) -> (Mode, Bandwidth) {
    // Future: Voip + low bitrate → SILK-only (NB/MB/WB); mid-rate SWB/FB speech → Hybrid.
    // Today CELT-only serves every application; pick the widest bandwidth the intent wants.
    let bandwidth = match cfg.application {
        Application::Voip => Bandwidth::Wb, // speech band; a stand-in until SILK lands
        Application::Audio | Application::LowDelay => Bandwidth::Fb,
    };
    (Mode::CeltOnly, bandwidth)
}

/// The constant CBR payload budget (bytes after the TOC) for a target bitrate and frame duration
/// (RFC 6716 §2.1.4 / §3.2.5). `bitrate · (frame_ms_tenths/10000) s / 8` bytes total, minus the
/// one §3.1 TOC byte, clamped to the §3.2.1 `2..=1275` frame range.
fn cbr_payload_bytes(bitrate_bps: u32, frame_ms_tenths: u16) -> usize {
    // total_bytes = bitrate_bps * frame_seconds / 8
    //             = bitrate_bps * (frame_ms_tenths / 10_000) / 8
    //             = bitrate_bps * frame_ms_tenths / 80_000
    let total_bytes = (u64::from(bitrate_bps) * u64::from(frame_ms_tenths) / 80_000) as usize;
    total_bytes.saturating_sub(1).clamp(2, MAX_FRAME_BYTES)
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // tests: one-shot fixtures, not a hot path
mod tests {
    use super::*;
    use oxideav_opus::OpusDecoder;

    /// SNR (dB) between a reference and a candidate over the overlap, in `i16` units.
    fn snr(reference: &[i16], test: &[i16]) -> f64 {
        let n = reference.len().min(test.len());
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for i in 0..n {
            let r = reference[i] as f64;
            let e = r - test[i] as f64;
            sig += r * r;
            err += e * e;
        }
        if err <= 0.0 {
            200.0
        } else {
            10.0 * (sig / err).log10()
        }
    }

    /// A 440 Hz + 1 kHz mono tone, `frames` frames of `n` samples each, amplitude ~0.3 FS.
    fn tone(n: usize, frames: usize) -> Vec<i16> {
        let mut v = Vec::with_capacity(n * frames);
        for i in 0..n * frames {
            let t = i as f64 / OPUS_RATE as f64;
            let s = 0.3 * ((2.0 * std::f64::consts::PI * 440.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin());
            v.push((s * 16384.0) as i16);
        }
        v
    }

    #[test]
    fn cbr_budget_matches_bitrate() {
        // 96 kbps, 20 ms → 240 total bytes → 239 payload.
        assert_eq!(cbr_payload_bytes(96_000, 200), 239);
        // 64 kbps, 20 ms → 160 total → 159 payload.
        assert_eq!(cbr_payload_bytes(64_000, 200), 159);
        // Tiny bitrate clamps up to the 2-byte floor.
        assert_eq!(cbr_payload_bytes(1, 200), 2);
        // Huge bitrate clamps to the 1275 ceiling.
        assert_eq!(cbr_payload_bytes(10_000_000, 200), MAX_FRAME_BYTES);
    }

    #[test]
    fn rejects_bad_configs() {
        assert!(OpusEncoder::new(EncoderConfig { sample_rate: 44_100, ..EncoderConfig::new(2) }).is_err());
        assert!(OpusEncoder::new(EncoderConfig { channels: 3, ..EncoderConfig::new(2) }).is_err());
        assert!(OpusEncoder::new(EncoderConfig { frame_ms_tenths: 400, ..EncoderConfig::new(2) }).is_err());
    }

    /// End-to-end: encode a tone through the CELT-only encoder, decode it back through our own
    /// `OpusDec`, and confirm the reconstruction tracks the input. CELT is lossy + float, so we
    /// gate on a modest SNR over the steady body (past the 2.5 ms MDCT-overlap startup).
    #[test]
    fn mono_tone_round_trips_through_our_decoder() {
        let mut enc = OpusEncoder::new(EncoderConfig::new(1)).unwrap();
        let n = enc.frame_samples();
        let frames = 50;
        let input = tone(n, frames);

        let mut dec = OpusDecoder::new();
        let mut out = Vec::new();
        let mut packet = Vec::new();
        for f in 0..frames {
            enc.encode_frame(&input[f * n..(f + 1) * n], &mut packet).unwrap();
            let ch = dec.decode_packet(&packet).unwrap();
            assert_eq!(ch.samples_per_channel(), n);
            out.extend_from_slice(&ch.pcm);
        }

        // Drop the first two frames on each side to skip the encoder+decoder startup transient,
        // then align: CELT's encode+decode chain delays by 120 samples (2.5 ms MDCT overlap).
        const DELAY: usize = 120;
        let start = 2 * n;
        let s = snr(&input[start..], &out[start + DELAY..]);
        assert!(s > 12.0, "mono CELT round-trip SNR {s:.1} dB too low");
    }

    #[test]
    fn stereo_tone_round_trips_through_our_decoder() {
        let mut enc = OpusEncoder::new(EncoderConfig::new(2)).unwrap();
        let n = enc.frame_samples();
        let frames = 50;
        // Interleave the same tone on both channels.
        let mono = tone(n, frames);
        let mut input = Vec::with_capacity(mono.len() * 2);
        for &s in &mono {
            input.push(s);
            input.push(s);
        }

        let mut dec = OpusDecoder::new();
        let mut out = Vec::new();
        let mut packet = Vec::new();
        for f in 0..frames {
            enc.encode_frame(&input[f * n * 2..(f + 1) * n * 2], &mut packet).unwrap();
            let ch = dec.decode_packet(&packet).unwrap();
            assert_eq!(ch.samples_per_channel(), n);
            out.extend_from_slice(&ch.pcm);
        }
        const DELAY: usize = 120;
        let start = 2 * n * 2;
        let s = snr(&input[start..], &out[start + DELAY * 2..]);
        assert!(s > 12.0, "stereo CELT round-trip SNR {s:.1} dB too low");
    }

    #[test]
    fn digital_silence_round_trips_to_silence() {
        let mut enc = OpusEncoder::new(EncoderConfig::new(1)).unwrap();
        let n = enc.frame_samples();
        let input = vec![0i16; n];
        let mut dec = OpusDecoder::new();
        let mut packet = Vec::new();
        enc.encode_frame(&input, &mut packet).unwrap();
        let ch = dec.decode_packet(&packet).unwrap();
        assert!(ch.pcm.iter().all(|&v| v == 0), "silence must decode to silence");
    }
}
