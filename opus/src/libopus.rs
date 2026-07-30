//! libopus encoder backend for [`crate::opusenc::OpusEnc`] — a safe RAII wrapper over the vendored,
//! statically-linked reference **libopus** C encoder ([`libopus_sys`]), used when the `libopus`
//! feature is on (the default). libopus is the RFC 6716 reference encoder: reference-grade quality
//! and allocation-free on the hot path (C99 VLA scratch), so a steady encode adds no heap traffic.
//!
//! Everything touching the C `OpusEncoder` is `unsafe`; this module is the safe boundary.

use std::os::raw::c_int;

use libopus_sys as sys;

use crate::encoder::{Application, EncoderConfig, OPUS_RATE};

/// Safe owner of a C `OpusEncoder`: interleaved 48 kHz `i16` frames in, Opus packets out.
pub struct LibopusEncoder {
    st: *mut sys::OpusEncoder,
    channels: usize,
    /// Samples per channel per frame (a valid Opus frame size, e.g. 960 for 20 ms @ 48 kHz).
    frame_samples: usize,
}

// SAFETY: the C encoder struct is owned solely here and only ever accessed through `&mut self`
// (opus_encode / opus_encoder_ctl require exclusive access). The element holding it is run by one
// scheduler thread at a time (moved between threads, never shared), so there is no concurrent
// access — `Send` is sound. It is deliberately not `Sync`.
unsafe impl Send for LibopusEncoder {}

impl LibopusEncoder {
    /// Build a reference-libopus encoder for `cfg` (48 kHz, 1–2 channels). Errors with the
    /// libopus message on an unsupported config or a create failure.
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
        let mut me = Self { st, channels, frame_samples };
        // CBR at the target bitrate (matches the pure-Rust backend's default rate control).
        me.ctl_set(sys::OPUS_SET_BITRATE_REQUEST, cfg.bitrate_bps as i32);
        Ok(me)
    }

    fn ctl_set(&mut self, request: c_int, value: i32) {
        // SAFETY: every `SET_*` request takes exactly one `i32` vararg.
        unsafe { sys::opus_encoder_ctl(self.st, request, value) };
    }

    /// Total interleaved `i16` values one [`Self::encode_frame`] consumes (`channels · frame`).
    pub fn frame_len(&self) -> usize {
        self.channels * self.frame_samples
    }

    /// Encode one frame of interleaved 48 kHz `i16` PCM into `out` (its previous contents replaced).
    /// `out`'s capacity is reused across calls, so a steady encode does no allocation on this side,
    /// and libopus itself allocates nothing on the hot path.
    pub fn encode_frame(&mut self, pcm: &[i16], out: &mut Vec<u8>) -> Result<(), String> {
        if pcm.len() != self.frame_len() {
            return Err(format!("frame is {} samples, expected {}", pcm.len(), self.frame_len()));
        }
        // Headroom over the §3.2.1 1275-byte max frame (VBR bursts / future multi-frame). Reused.
        const CAP: usize = 4000;
        out.clear();
        out.reserve(CAP);
        // SAFETY: `out` has ≥ CAP bytes reserved; `pcm` holds channels·frame_samples samples;
        // `frame_samples` is a valid Opus frame size; opus_encode writes `n ≤ CAP` bytes.
        let n = unsafe {
            sys::opus_encode(self.st, pcm.as_ptr(), self.frame_samples as c_int, out.as_mut_ptr(), CAP as i32)
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

/// The libopus message for an error code.
fn strerror(code: c_int) -> String {
    // SAFETY: opus_strerror returns a static NUL-terminated string for any code.
    unsafe { std::ffi::CStr::from_ptr(sys::opus_strerror(code)) }
        .to_string_lossy()
        .into_owned()
}
