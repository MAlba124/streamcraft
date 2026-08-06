//! `libopus-sys` — minimal static FFI to the vendored libopus (RFC 6716) C reference codec.
//!
//! The C source under `vendor/opus/` (libopus 1.4, BSD-3-Clause) is compiled to a static archive by
//! `build.rs` (`cc`, no cmake/system lib) and linked in — so this is a self-contained, statically
//! linked Opus. Only the encoder/decoder surface `pf-opus` needs is bound (create/encode/decode/
//! ctl/destroy + the relevant constants), hand-written rather than via bindgen to avoid a libclang
//! build dependency. libopus is allocation-free on the hot path (C99 VLA scratch; `VAR_ARRAYS`).
//!
//! Everything here is `unsafe extern "C"` — the safe wrappers live in `pf-opus`.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int};

/// Opaque `OpusEncoder` (its size is runtime-determined via [`opus_encoder_get_size`]; always
/// heap-behind-a-pointer through [`opus_encoder_create`]).
#[repr(C)]
pub struct OpusEncoder {
    _private: [u8; 0],
}

/// Opaque `OpusDecoder`.
#[repr(C)]
pub struct OpusDecoder {
    _private: [u8; 0],
}

// ---- Return / status codes (opus_defines.h) ----
pub const OPUS_OK: c_int = 0;

// ---- Applications (encoder mode preset) ----
pub const OPUS_APPLICATION_VOIP: c_int = 2048;
pub const OPUS_APPLICATION_AUDIO: c_int = 2049;
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: c_int = 2051;

// ---- Signal hints ----
pub const OPUS_SIGNAL_VOICE: c_int = 3001;
pub const OPUS_SIGNAL_MUSIC: c_int = 3002;

// ---- Special values ----
pub const OPUS_AUTO: c_int = -1000;
pub const OPUS_BITRATE_MAX: c_int = -1;

// ---- CTL requests (encoder unless noted) ----
pub const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
pub const OPUS_GET_BITRATE_REQUEST: c_int = 4003;
pub const OPUS_SET_VBR_REQUEST: c_int = 4006;
pub const OPUS_SET_COMPLEXITY_REQUEST: c_int = 4010;
pub const OPUS_SET_INBAND_FEC_REQUEST: c_int = 4012;
pub const OPUS_SET_PACKET_LOSS_PERC_REQUEST: c_int = 4014;
pub const OPUS_SET_VBR_CONSTRAINT_REQUEST: c_int = 4020;
pub const OPUS_SET_SIGNAL_REQUEST: c_int = 4024;
/// Shared by encoder + decoder: reset all state to a freshly-initialized stream start.
pub const OPUS_RESET_STATE: c_int = 4028;

extern "C" {
    /// `"libopus <version>"`, NUL-terminated static string.
    pub fn opus_get_version_string() -> *const c_char;
    /// Human-readable message for an error code, NUL-terminated static string.
    pub fn opus_strerror(error: c_int) -> *const c_char;

    // ---- Encoder ----
    pub fn opus_encoder_get_size(channels: c_int) -> c_int;
    pub fn opus_encoder_create(
        fs: i32,
        channels: c_int,
        application: c_int,
        error: *mut c_int,
    ) -> *mut OpusEncoder;
    /// Encode interleaved `i16` PCM (`frame_size` samples per channel) into `data`; returns the
    /// packet length in bytes or a negative error code.
    pub fn opus_encode(
        st: *mut OpusEncoder,
        pcm: *const i16,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    /// `opus_encode` for interleaved `f32` PCM (nominally `[-1, 1]`).
    pub fn opus_encode_float(
        st: *mut OpusEncoder,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;
    /// Variadic control interface. For the `SET_*` requests the trailing arg is one `i32`; for
    /// `GET_*` it is a `*mut i32`. Returns `OPUS_OK` or a negative error.
    pub fn opus_encoder_ctl(st: *mut OpusEncoder, request: c_int, ...) -> c_int;
    pub fn opus_encoder_destroy(st: *mut OpusEncoder);

    /// Channel count (1 or 2) of an Opus packet, read from its TOC byte. Negative on a bad packet.
    pub fn opus_packet_get_nb_channels(data: *const u8) -> c_int;

    // ---- Decoder ----
    pub fn opus_decoder_get_size(channels: c_int) -> c_int;
    pub fn opus_decoder_create(fs: i32, channels: c_int, error: *mut c_int) -> *mut OpusDecoder;
    /// Decode `data` into interleaved `i16` PCM; returns samples-per-channel or a negative error.
    /// `decode_fec` non-zero requests forward-error-correction reconstruction.
    pub fn opus_decode(
        st: *mut OpusDecoder,
        data: *const u8,
        len: i32,
        pcm: *mut i16,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_decode_float(
        st: *mut OpusDecoder,
        data: *const u8,
        len: i32,
        pcm: *mut f32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
    pub fn opus_decoder_ctl(st: *mut OpusDecoder, request: c_int, ...) -> c_int;
    pub fn opus_decoder_destroy(st: *mut OpusDecoder);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vendored library links statically and reports a version — proves the whole C build +
    /// FFI is wired.
    #[test]
    fn version_string_links() {
        // SAFETY: returns a static NUL-terminated string.
        let v = unsafe { std::ffi::CStr::from_ptr(opus_get_version_string()) };
        let s = v.to_str().unwrap();
        assert!(s.contains("libopus"), "unexpected version string: {s:?}");
        println!("linked: {s}");
    }

    /// End-to-end through the real C encoder + decoder: create a stereo 48 kHz encoder, set a
    /// bitrate, encode a 20 ms tone frame, decode it back, and confirm the frame size round-trips.
    /// This exercises `opus_encode`, `opus_encoder_ctl` (variadic), and `opus_decode`.
    #[test]
    fn encode_decode_roundtrip() {
        const FS: i32 = 48_000;
        const CH: c_int = 2;
        const FRAME: c_int = 960; // 20 ms @ 48 kHz

        unsafe {
            let mut err: c_int = 0;
            let enc = opus_encoder_create(FS, CH, OPUS_APPLICATION_AUDIO, &mut err);
            assert_eq!(err, OPUS_OK, "encoder_create: {}", strerror(err));
            assert!(!enc.is_null());
            assert_eq!(opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, 96_000i32), OPUS_OK);

            // A 440 Hz tone, interleaved stereo i16.
            let mut pcm = vec![0i16; (FRAME as usize) * CH as usize];
            for i in 0..FRAME as usize {
                let t = i as f64 / FS as f64;
                let s = (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 16384.0) as i16;
                pcm[i * 2] = s;
                pcm[i * 2 + 1] = s;
            }

            let mut packet = vec![0u8; 4000];
            let n = opus_encode(enc, pcm.as_ptr(), FRAME, packet.as_mut_ptr(), packet.len() as i32);
            assert!(n > 1, "encode returned {n} ({})", strerror(n));
            println!("encoded 20ms stereo @96k → {n} bytes");

            let dec = opus_decoder_create(FS, CH, &mut err);
            assert_eq!(err, OPUS_OK, "decoder_create: {}", strerror(err));
            let mut out = vec![0i16; (FRAME as usize) * CH as usize];
            let got = opus_decode(dec, packet.as_ptr(), n, out.as_mut_ptr(), FRAME, 0);
            assert_eq!(got, FRAME, "decoded {got} samples/ch ({})", strerror(got));

            opus_encoder_destroy(enc);
            opus_decoder_destroy(dec);
        }
    }

    fn strerror(code: c_int) -> String {
        // SAFETY: opus_strerror returns a static NUL-terminated string for any code.
        unsafe { std::ffi::CStr::from_ptr(opus_strerror(code)) }
            .to_string_lossy()
            .into_owned()
    }
}
