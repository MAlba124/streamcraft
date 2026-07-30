//! libopus encoder backend for [`crate::opusenc::OpusEnc`] — a safe RAII wrapper over the vendored,
//! statically-linked reference **libopus** C encoder ([`libopus_sys`]), used when the `libopus`
//! feature is on (the default). libopus is the RFC 6716 reference encoder: reference-grade quality
//! and allocation-free on the hot path (C99 VLA scratch), so a steady encode adds no heap traffic.
//!
//! Everything touching the C `OpusEncoder` is `unsafe`; this module is the safe boundary.

use std::os::raw::c_int;

use libopus_sys as sys;

use crate::encoder::{Application, EncoderConfig, Signal, OPUS_RATE};

/// Safe owner of a C `OpusEncoder`: interleaved 48 kHz `i16` frames in, Opus packets out.
pub struct LibopusEncoder {
    st: *mut sys::OpusEncoder,
    channels: usize,
    /// Samples per channel per frame (a valid Opus frame size, e.g. 960 for 20 ms @ 48 kHz).
    frame_samples: usize,
    /// Fallback aligned copy for a misaligned input frame (whole-sample upstreams like flacdec are
    /// 2-aligned, so this stays empty; a `bytes`-bridge source that splits a sample could hit it).
    aligned: Vec<i16>,
}

// SAFETY: the C encoder struct is owned solely here and only ever accessed through `&mut self`
// (opus_encode / opus_encoder_ctl require exclusive access). The element holding it is run by one
// scheduler thread at a time (moved between threads, never shared), so there is no concurrent
// access — `Send` is sound. It is deliberately not `Sync`.
unsafe impl Send for LibopusEncoder {}

impl LibopusEncoder {
    /// Build a reference-libopus encoder for `cfg` (48 kHz, 1–2 channels). Errors with the
    /// libopus message on an unsupported config or a create failure.
    #[allow(clippy::disallowed_methods)] // one-time per-stream construction; `aligned` is a reused fallback scratch
    pub fn new(cfg: EncoderConfig) -> Result<Self, String> {
        if cfg.sample_rate != OPUS_RATE {
            return Err(format!("libopus backend requires 48 kHz, got {}", cfg.sample_rate));
        }
        let channels = cfg.channels as usize;
        if !(1..=2).contains(&channels) {
            return Err(format!("libopus backend supports 1–2 channels, got {channels}"));
        }
        let application = match cfg.application {
            Application::Voip => sys::OPUS_APPLICATION_VOIP,
            Application::Audio => sys::OPUS_APPLICATION_AUDIO,
            Application::LowDelay => sys::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
        };

        let mut err: c_int = 0;
        // SAFETY: standard opus_encoder_create; `err` receives the status, `st` the handle.
        let st = unsafe {
            sys::opus_encoder_create(cfg.sample_rate as i32, channels as c_int, application, &mut err)
        };
        if err != sys::OPUS_OK || st.is_null() {
            return Err(format!("opus_encoder_create failed: {}", strerror(err)));
        }

        let frame_samples = (cfg.sample_rate as usize * cfg.frame_ms_tenths as usize) / 10_000;
        let mut me = Self { st, channels, frame_samples, aligned: Vec::new() };
        // The rate-distortion surface (see EncoderConfig). libopus defaults to VBR + high
        // complexity; we set them explicitly so the config is authoritative and searchable.
        me.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, cfg.bitrate_bps as i32);
        me.ctl_set(sys::OPUS_SET_VBR_REQUEST, i32::from(cfg.vbr));
        me.ctl_set(sys::OPUS_SET_VBR_CONSTRAINT_REQUEST, i32::from(cfg.vbr_constrained));
        me.ctl_set(sys::OPUS_SET_COMPLEXITY_REQUEST, i32::from(cfg.complexity.min(10)));
        me.ctl_set(
            sys::OPUS_SET_SIGNAL_REQUEST,
            match cfg.signal {
                Signal::Auto => sys::OPUS_AUTO,
                Signal::Voice => sys::OPUS_SIGNAL_VOICE,
                Signal::Music => sys::OPUS_SIGNAL_MUSIC,
            },
        );
        Ok(me)
    }

    fn ctl_set(&mut self, request: c_int, value: i32) {
        // SAFETY: every `SET_*` request takes exactly one `i32` vararg.
        unsafe { sys::opus_encoder_ctl(self.st, request, value) };
    }

    /// Bytes of interleaved `s16` PCM one [`Self::encode`] consumes (`channels · frame · 2`).
    pub fn frame_bytes(&self) -> usize {
        self.channels * self.frame_samples * 2
    }

    /// Encode one frame of interleaved 48 kHz **`s16`-LE PCM bytes** into `out` (contents replaced).
    ///
    /// Zero-copy on the common path: on little-endian, `s16`-LE bytes *are* `i16`, so a 2-aligned
    /// `frame` is handed straight to libopus with no conversion or copy (flacdec and the audio
    /// elements emit whole 2-aligned samples). A misaligned frame (a `bytes` source that split a
    /// sample) is copied once into a reused aligned buffer. `out`'s capacity is reused, and libopus
    /// allocates nothing on the hot path, so a steady encode does no heap traffic.
    pub fn encode(&mut self, frame: &[u8], out: &mut Vec<u8>) -> Result<(), String> {
        if frame.len() != self.frame_bytes() {
            return Err(format!("frame is {} bytes, expected {}", frame.len(), self.frame_bytes()));
        }
        // Headroom over the §3.2.1 1275-byte max frame (VBR bursts / future multi-frame). Reused.
        const CAP: usize = 4000;
        out.clear();
        out.reserve(CAP);

        // s16-LE bytes reinterpreted as i16 (little-endian target). Aligned → in place; else copy.
        #[cfg(target_endian = "big")]
        compile_error!("libopus s16 reinterpret assumes a little-endian target");
        let pcm: *const i16 = if (frame.as_ptr() as usize).is_multiple_of(2) {
            frame.as_ptr().cast::<i16>()
        } else {
            self.aligned.clear();
            let (pairs, _) = frame.as_chunks::<2>();
            self.aligned.extend(pairs.iter().map(|c| i16::from_le_bytes(*c)));
            self.aligned.as_ptr()
        };

        // SAFETY: `out` has ≥ CAP bytes reserved; `pcm` points at `frame_samples · channels`
        // 2-aligned `i16` values (in-place aligned frame or the aligned copy); `frame_samples` is a
        // valid Opus frame size; opus_encode writes `n ≤ CAP` bytes.
        let n = unsafe {
            sys::opus_encode(self.st, pcm, self.frame_samples as c_int, out.as_mut_ptr(), CAP as i32)
        };
        if n < 0 {
            return Err(format!("opus_encode: {}", strerror(n)));
        }
        // SAFETY: opus_encode wrote exactly `n` initialized bytes into the reserved buffer.
        unsafe { out.set_len(n as usize) };
        Ok(())
    }

    /// Reset all carried inter-frame state (stream start / after a seek).
    pub fn reset(&mut self) {
        // SAFETY: OPUS_RESET_STATE consumes no vararg.
        unsafe { sys::opus_encoder_ctl(self.st, sys::OPUS_RESET_STATE) };
    }
}

impl Drop for LibopusEncoder {
    fn drop(&mut self) {
        // SAFETY: `st` came from opus_encoder_create and is freed exactly once here.
        unsafe { sys::opus_encoder_destroy(self.st) };
    }
}

/// Maximum samples per channel one Opus packet can decode to (120 ms @ 48 kHz — the §3.2.5 limit).
const MAX_FRAME_SAMPLES: usize = 5760;

/// Safe owner of a C `OpusDecoder`: Opus packets in, interleaved 48 kHz `i16` PCM out.
///
/// The output channel count is fixed when the decoder is created — from an `OpusHead` hint
/// ([`Self::set_channels`]) when the container front-loads it, else auto-detected from the first
/// packet's TOC. A fixed output width is what makes libopus transparently up/down-mix a mono-coded
/// frame inside a stereo stream (the pure-Rust decoder reports per-packet channels instead).
pub struct LibopusDecoder {
    st: *mut sys::OpusDecoder,
    channels: usize,
    /// `OpusHead` channel hint (0 = none yet); used when the decoder is lazily created.
    hint: usize,
}

// SAFETY: as for LibopusEncoder — the C decoder is owned solely here, accessed only through
// `&mut self`, and the element runs on one scheduler thread at a time.
unsafe impl Send for LibopusDecoder {}

impl Default for LibopusDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl LibopusDecoder {
    pub fn new() -> Self {
        Self { st: std::ptr::null_mut(), channels: 0, hint: 0 }
    }

    /// Fix the output channel count from an `OpusHead` (RFC 7845) before the first packet.
    pub fn set_channels(&mut self, channels: usize) {
        if self.st.is_null() && (1..=2).contains(&channels) {
            self.hint = channels;
        }
    }

    /// Decode one packet into `pcm` (interleaved `s16`, contents replaced); returns the output
    /// channel count. `pcm`'s capacity is reused, so a steady decode does no allocation on this
    /// side, and libopus itself is allocation-free on the hot path.
    pub fn decode_packet_into(&mut self, packet: &[u8], pcm: &mut Vec<i16>) -> Result<u8, String> {
        if self.st.is_null() {
            let channels = if self.hint != 0 {
                self.hint as c_int
            } else if packet.is_empty() {
                return Err("empty packet before channel count is known".into());
            } else {
                // SAFETY: `packet` is non-empty; the function only reads the TOC byte.
                let ch = unsafe { sys::opus_packet_get_nb_channels(packet.as_ptr()) };
                if ch < 1 {
                    return Err(format!("opus_packet_get_nb_channels: {}", strerror(ch)));
                }
                ch
            };
            let mut err: c_int = 0;
            // SAFETY: standard opus_decoder_create at Opus's fixed 48 kHz output rate.
            let st = unsafe { sys::opus_decoder_create(48_000, channels, &mut err) };
            if err != sys::OPUS_OK || st.is_null() {
                return Err(format!("opus_decoder_create failed: {}", strerror(err)));
            }
            self.st = st;
            self.channels = channels as usize;
        }

        pcm.clear();
        pcm.reserve(self.channels * MAX_FRAME_SAMPLES);
        // SAFETY: `pcm` has room for channels·MAX_FRAME_SAMPLES i16; opus_decode writes
        // `channels · got` (got ≤ MAX_FRAME_SAMPLES) initialized samples and returns `got`.
        let got = unsafe {
            sys::opus_decode(
                self.st,
                packet.as_ptr(),
                packet.len() as i32,
                pcm.as_mut_ptr(),
                MAX_FRAME_SAMPLES as c_int,
                0,
            )
        };
        if got < 0 {
            return Err(format!("opus_decode: {}", strerror(got)));
        }
        // SAFETY: opus_decode wrote `channels · got` initialized i16 samples.
        unsafe { pcm.set_len(self.channels * got as usize) };
        Ok(self.channels as u8)
    }

    /// Reset carried inter-packet state (seek / stream restart).
    pub fn reset(&mut self) {
        if !self.st.is_null() {
            // SAFETY: OPUS_RESET_STATE consumes no vararg.
            unsafe { sys::opus_decoder_ctl(self.st, sys::OPUS_RESET_STATE) };
        }
    }
}

impl Drop for LibopusDecoder {
    fn drop(&mut self) {
        if !self.st.is_null() {
            // SAFETY: `st` came from opus_decoder_create and is freed exactly once.
            unsafe { sys::opus_decoder_destroy(self.st) };
        }
    }
}

/// The libopus message for an error code.
fn strerror(code: c_int) -> String {
    // SAFETY: opus_strerror returns a static NUL-terminated string for any code.
    unsafe { std::ffi::CStr::from_ptr(sys::opus_strerror(code)) }
        .to_string_lossy()
        .into_owned()
}
