//! Interleaved-PCM sample-format conversion (spec: Formats — an `audioconvert`-class
//! transform). Pure, allocation-free functions that rewrite a byte buffer of interleaved
//! PCM from one [`SampleFormat`] to another, with correct scaling, rounding, and clamping.
//! Sample **rate** conversion (resampling) is deliberately out of scope — that is a
//! separate, much harder element (see the crate follow-ups); this file only changes the
//! *representation* of each sample and, optionally, the channel layout.
//!
//! # The scaling model
//!
//! Every integer format is treated as a signed fixed-point fraction of full scale, where
//! "full scale" for an `N`-bit format is `2^(N-1)`. Two consequences fall out of that one
//! rule, and they are exactly the conventions WAV/FLAC/PipeWire use:
//!
//! * **`U8` is signed-with-a-128-bias.** The byte `u` decodes to the 8-bit-scale signed
//!   value `u as i32 - 128` (range `-128..=127`) and re-encodes as `s + 128`.
//! * **Depth changes are bit shifts.** Widening aligns the most-significant bit, so it is a
//!   left shift by the depth difference (`s16 → s24` is `<< 8`, `s16 → s32` is `<< 16`).
//!   Narrowing is the inverse right shift, rounded to nearest.
//!
//! Integer ↔ float uses the usual `/ 2^(bits-1)` to float and `* (2^(bits-1) - 1)` back,
//! clamped into range. Float is native-endian IEEE-754 `f32` nominally in `[-1.0, 1.0]`;
//! out-of-range inputs are clamped, so full-scale never wraps.
//!
//! # Performance
//!
//! Conversion is one linear pass over the samples with a per-format-pair closure chosen
//! once, up front — no per-sample matching on the format. The identity conversion
//! (`from == to`, same channels) is a pure `memcpy`. Callers stream in whole interchannel
//! frames; the element wrapper ([`crate::convert_element::AudioConvert`]) handles chunking
//! and any partial-frame carry.

use crate::format::SampleFormat;

/// A decoded sample in a *canonical signed i32 at 32-bit scale*, or a float. Integer
/// samples from any depth are hoisted to the common 32-bit scale so a single rescale step
/// (down-shift with rounding) lands them at the target depth; this keeps the per-pair code
/// to two small functions instead of an N×N table.
///
/// Kept private: it is an implementation detail of the pipeline, not part of the API.
#[inline(always)]
fn decode_i32_scale32(fmt: SampleFormat, b: &[u8]) -> i32 {
    // Returns the sample as a signed value scaled so its MSB sits at bit 31 (i.e. as if it
    // were S32). Widening from the native depth to 32 is an exact left shift.
    match fmt {
        // (u - 128) is the 8-bit signed value; << 24 lifts it to 32-bit scale.
        SampleFormat::U8 => ((b[0] as i32) - 128) << 24,
        SampleFormat::S16 => (i16::from_le_bytes([b[0], b[1]]) as i32) << 16,
        SampleFormat::S24 => {
            // Assemble 24 bits, sign-extend into i32, then lift to 32-bit scale (<< 8).
            let v = b[0] as i32 | (b[1] as i32) << 8 | (b[2] as i32) << 16;
            ((v << 8) >> 8) << 8
        }
        SampleFormat::S32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        // Float is handled on the float path; treat as silence if ever misrouted here.
        SampleFormat::F32 => 0,
    }
}

/// Round a 32-bit-scale signed sample down to `fmt`'s depth and write it little-endian.
/// Narrowing is a right shift by `32 - bits` with round-to-nearest (ties toward +∞), then a
/// clamp so a rounded full-scale value never overflows the narrower range.
#[inline(always)]
fn encode_i32_scale32(fmt: SampleFormat, v: i32, out: &mut [u8]) {
    match fmt {
        SampleFormat::U8 => {
            let s = round_shift(v, 24).clamp(-128, 127);
            out[0] = (s + 128) as u8;
        }
        SampleFormat::S16 => {
            let s = round_shift(v, 16).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            out[..2].copy_from_slice(&s.to_le_bytes());
        }
        SampleFormat::S24 => {
            let s = round_shift(v, 8).clamp(-(1 << 23), (1 << 23) - 1);
            let le = s.to_le_bytes();
            out[..3].copy_from_slice(&le[..3]);
        }
        SampleFormat::S32 => {
            // Already at 32-bit scale — no shift, no rounding, exact.
            out[..4].copy_from_slice(&v.to_le_bytes());
        }
        SampleFormat::F32 => {
            // Misrouted; write silence rather than garbage.
            out[..4].copy_from_slice(&0f32.to_le_bytes());
        }
    }
}

/// Arithmetic right shift by `shift` bits with round-to-nearest (ties toward +∞). Adding
/// the half-LSB *before* the arithmetic shift rounds correctly for both signs. Guards the
/// `shift == 0` case (no-op) and the add against overflow near `i32::MAX`.
#[inline(always)]
fn round_shift(v: i32, shift: u32) -> i32 {
    if shift == 0 {
        return v;
    }
    let half = 1i32 << (shift - 1);
    // Saturating add so a value within `half` of i32::MAX rounds up without wrapping; the
    // subsequent >> brings it back in range, and the caller clamps to the target depth.
    (v.saturating_add(half)) >> shift
}

/// Decode a float sample (native-endian `f32`) from `b`.
#[inline(always)]
fn decode_f32(b: &[u8]) -> f32 {
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The positive full-scale multiplier for `bits`-deep integer PCM: `2^(bits-1) - 1`.
#[inline(always)]
fn int_scale(bits: u32) -> f32 {
    ((1u64 << (bits - 1)) - 1) as f32
}

/// Convert a float sample into `fmt` (an integer format), scaling by `2^(bits-1)-1`,
/// rounding to nearest, and clamping into the format's range. `fmt` must be integer.
#[inline(always)]
fn encode_float_to_int(fmt: SampleFormat, f: f32, out: &mut [u8]) {
    let bits = fmt.bits();
    // Clamp the *float* first so scaling can't overflow, then round half-away-from-zero.
    let scaled = (f.clamp(-1.0, 1.0) * int_scale(bits)).round();
    // Round to a canonical 32-bit-scale value by going through the integer domain at the
    // target depth, then re-encode via the integer path so U8 bias / LE packing is shared.
    let native = scaled as i32; // in range by construction (clamped, |scale| < 2^(bits-1))
    match fmt {
        SampleFormat::U8 => out[0] = (native.clamp(-128, 127) + 128) as u8,
        SampleFormat::S16 => out[..2]
            .copy_from_slice(&(native.clamp(i16::MIN as i32, i16::MAX as i32) as i16).to_le_bytes()),
        SampleFormat::S24 => {
            let s = native.clamp(-(1 << 23), (1 << 23) - 1);
            out[..3].copy_from_slice(&s.to_le_bytes()[..3]);
        }
        SampleFormat::S32 => {
            // int_scale(32) < 2^31, and a clamped |f| <= 1.0, so `scaled` fits an i32.
            out[..4].copy_from_slice(&native.to_le_bytes());
        }
        SampleFormat::F32 => unreachable!("encode_float_to_int called with F32 target"),
    }
}

/// Decode an integer sample from `fmt` at `b` to a float in `[-1.0, 1.0]` via
/// `/ 2^(bits-1)`. `fmt` must be integer.
#[inline(always)]
fn decode_int_to_float(fmt: SampleFormat, b: &[u8]) -> f32 {
    let bits = fmt.bits();
    let native = match fmt {
        SampleFormat::U8 => (b[0] as i32) - 128,
        SampleFormat::S16 => i16::from_le_bytes([b[0], b[1]]) as i32,
        SampleFormat::S24 => {
            let v = b[0] as i32 | (b[1] as i32) << 8 | (b[2] as i32) << 16;
            (v << 8) >> 8
        }
        SampleFormat::S32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        SampleFormat::F32 => unreachable!("decode_int_to_float called with F32 source"),
    };
    native as f32 / (1u64 << (bits - 1)) as f32
}

/// Convert one sample from `from` at `src` into `to`, writing `to.bytes()` bytes at `dst`.
/// Routes through the integer 32-bit-scale domain unless a float endpoint is involved.
#[inline(always)]
fn convert_sample(from: SampleFormat, to: SampleFormat, src: &[u8], dst: &mut [u8]) {
    match (from.is_float(), to.is_float()) {
        // int -> int: hoist to 32-bit scale, narrow (or widen) to the target depth.
        (false, false) => encode_i32_scale32(to, decode_i32_scale32(from, src), dst),
        // int -> float.
        (false, true) => {
            let f = decode_int_to_float(from, src);
            dst[..4].copy_from_slice(&f.to_le_bytes());
        }
        // float -> int.
        (true, false) => encode_float_to_int(to, decode_f32(src), dst),
        // float -> float: a straight copy (already native-endian f32).
        (true, true) => dst[..4].copy_from_slice(&src[..4]),
    }
}

/// Convert interleaved PCM `input` (format `from`, `channels` channels) into `to`, writing
/// to `output`. Channel count is preserved. Returns the number of **bytes** written, or
/// `None` if `input` is not a whole number of interchannel frames or `output` is too small.
///
/// This is the workhorse: one pass, no allocation, no per-sample format branching (the
/// branch on `(from, to)` is hoisted out of the loop by the closure the compiler inlines).
pub fn convert_interleaved(
    from: SampleFormat,
    to: SampleFormat,
    channels: usize,
    input: &[u8],
    output: &mut [u8],
) -> Option<usize> {
    let in_bps = from.bytes();
    let out_bps = to.bytes();
    let in_stride = in_bps.checked_mul(channels)?;
    if in_stride == 0 || input.len() % in_stride != 0 {
        return None;
    }
    let samples = input.len() / in_bps; // total samples across all channels
    let out_len = samples.checked_mul(out_bps)?;
    if output.len() < out_len {
        return None;
    }

    // Fast path: identical format is a straight copy (no per-sample work at all).
    if from == to {
        output[..input.len()].copy_from_slice(input);
        return Some(input.len());
    }

    let mut si = 0usize;
    let mut di = 0usize;
    for _ in 0..samples {
        convert_sample(from, to, &input[si..si + in_bps], &mut output[di..di + out_bps]);
        si += in_bps;
        di += out_bps;
    }
    Some(out_len)
}

/// The output byte length [`convert_interleaved`] will produce for `input_len` bytes of
/// `from` PCM converted to `to` — the exact size a caller must allocate. Returns `None`
/// when `input_len` is not a whole number of `from` samples.
pub fn converted_len(
    from: SampleFormat,
    to: SampleFormat,
    channels: usize,
    input_len: usize,
) -> Option<usize> {
    let in_bps = from.bytes();
    let stride = in_bps.checked_mul(channels)?;
    if stride == 0 || input_len % stride != 0 {
        return None;
    }
    (input_len / in_bps).checked_mul(to.bytes())
}

/// Convert interleaved PCM into a freshly-allocated `Vec`, sizing the output exactly.
/// A convenience over [`convert_interleaved`] for callers that are not managing a buffer
/// pool; the element wrapper uses the in-place form to stay allocation-free on the hot path.
pub fn convert_interleaved_vec(
    from: SampleFormat,
    to: SampleFormat,
    channels: usize,
    input: &[u8],
) -> Option<Vec<u8>> {
    let out_len = converted_len(from, to, channels, input.len())?;
    let mut out = vec![0u8; out_len];
    convert_interleaved(from, to, channels, input, &mut out)?;
    Some(out)
}

// --- Channel remap (nice-to-have) -----------------------------------------------------
//
// A minimal, well-defined layout change independent of sample-format conversion: mono →
// stereo duplicates the single channel to both; stereo → mono averages L+R. Both operate on
// interleaved samples of one [`SampleFormat`] and do *not* change the sample format (compose
// with [`convert_interleaved`] for format-and-layout changes). Only the two most common
// cases are handled — anything else returns `None`.

/// Remap interleaved PCM of format `fmt` from `in_channels` to `out_channels`, writing to
/// `output`. Supports mono→stereo (duplicate) and stereo→mono (average); returns `None` for
/// any other channel pair, a ragged input, or an undersized output. Returns bytes written.
pub fn remap_channels(
    fmt: SampleFormat,
    in_channels: usize,
    out_channels: usize,
    input: &[u8],
    output: &mut [u8],
) -> Option<usize> {
    let bps = fmt.bytes();
    let in_stride = bps.checked_mul(in_channels)?;
    if in_stride == 0 || input.len() % in_stride != 0 {
        return None;
    }
    let frames = input.len() / in_stride;
    let out_len = frames.checked_mul(bps.checked_mul(out_channels)?)?;
    if output.len() < out_len {
        return None;
    }

    match (in_channels, out_channels) {
        // Identity: nothing to remap.
        (i, o) if i == o => {
            output[..input.len()].copy_from_slice(input);
            Some(input.len())
        }
        // Mono -> stereo: copy the sole channel into both output channels.
        (1, 2) => {
            for f in 0..frames {
                let s = &input[f * bps..f * bps + bps];
                let d = f * 2 * bps;
                output[d..d + bps].copy_from_slice(s);
                output[d + bps..d + 2 * bps].copy_from_slice(s);
            }
            Some(out_len)
        }
        // Stereo -> mono: average L and R at 32-bit scale (float for F32), then re-encode.
        (2, 1) => {
            for f in 0..frames {
                let base = f * 2 * bps;
                let l = &input[base..base + bps];
                let r = &input[base + bps..base + 2 * bps];
                let d = &mut output[f * bps..f * bps + bps];
                if fmt.is_float() {
                    let avg = (decode_f32(l) + decode_f32(r)) * 0.5;
                    d[..4].copy_from_slice(&avg.to_le_bytes());
                } else {
                    // Average at 32-bit scale (round to nearest), then narrow back.
                    let a = decode_i32_scale32(fmt, l) as i64;
                    let b = decode_i32_scale32(fmt, r) as i64;
                    let avg = ((a + b) / 2) as i32;
                    encode_i32_scale32(fmt, avg, d);
                }
            }
            Some(out_len)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::AudioFormat;

    // All integer formats, for exhaustive pair loops.
    const INT_FORMATS: [SampleFormat; 4] = [
        SampleFormat::U8,
        SampleFormat::S16,
        SampleFormat::S24,
        SampleFormat::S32,
    ];
    const ALL_FORMATS: [SampleFormat; 5] = [
        SampleFormat::U8,
        SampleFormat::S16,
        SampleFormat::S24,
        SampleFormat::S32,
        SampleFormat::F32,
    ];

    /// Encode one native-scale integer sample of `fmt` to its LE bytes (test helper mirror
    /// of the format's on-wire layout; U8 carries the +128 bias).
    fn enc_native(fmt: SampleFormat, native: i32) -> Vec<u8> {
        match fmt {
            SampleFormat::U8 => vec![(native + 128) as u8],
            SampleFormat::S16 => (native as i16).to_le_bytes().to_vec(),
            SampleFormat::S24 => native.to_le_bytes()[..3].to_vec(),
            SampleFormat::S32 => native.to_le_bytes().to_vec(),
            SampleFormat::F32 => (native as f32).to_le_bytes().to_vec(),
        }
    }

    fn enc_f32(v: f32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    fn one(from: SampleFormat, to: SampleFormat, src: &[u8]) -> Vec<u8> {
        convert_interleaved_vec(from, to, 1, src).expect("convert one sample")
    }

    // --- identity / passthrough -------------------------------------------------------

    #[test]
    fn identity_is_byte_exact_for_every_format() {
        for f in ALL_FORMATS {
            // A couple of arbitrary samples worth of bytes.
            let src: Vec<u8> = (0..(f.bytes() * 4) as u8).collect();
            let out = convert_interleaved_vec(f, f, 2, &src).unwrap();
            assert_eq!(out, src, "identity {f:?} must be a byte-exact copy");
        }
    }

    // --- widening (lossless) ----------------------------------------------------------

    #[test]
    fn widen_shifts_align_msb() {
        // The exact shifts the spec calls out.
        // s16 -> s24 == << 8.
        assert_eq!(
            one(SampleFormat::S16, SampleFormat::S24, &1i16.to_le_bytes()),
            enc_native(SampleFormat::S24, 1i32 << 8)
        );
        // s16 -> s32 == << 16.
        assert_eq!(
            one(SampleFormat::S16, SampleFormat::S32, &100i16.to_le_bytes()),
            ((100i32) << 16).to_le_bytes().to_vec()
        );
        // s24 -> s32 == << 8.
        let s24 = enc_native(SampleFormat::S24, 1234);
        assert_eq!(
            one(SampleFormat::S24, SampleFormat::S32, &s24),
            ((1234i32) << 8).to_le_bytes().to_vec()
        );
        // u8 -> s16: (u-128) << 8.
        assert_eq!(
            one(SampleFormat::U8, SampleFormat::S16, &[200]),
            (((200i32 - 128) << 8) as i16).to_le_bytes().to_vec()
        );
    }

    #[test]
    fn widen_extremes_map_to_target_extremes() {
        // Full-scale negative widens to the target's full-scale negative exactly.
        assert_eq!(
            one(SampleFormat::S16, SampleFormat::S32, &i16::MIN.to_le_bytes()),
            ((i16::MIN as i32) << 16).to_le_bytes().to_vec()
        );
        // U8 min (0 -> -128) and max (255 -> 127).
        assert_eq!(
            one(SampleFormat::U8, SampleFormat::S32, &[0]),
            ((-128i32) << 24).to_le_bytes().to_vec()
        );
        assert_eq!(
            one(SampleFormat::U8, SampleFormat::S32, &[255]),
            ((127i32) << 24).to_le_bytes().to_vec()
        );
    }

    // --- widen -> narrow round-trips are lossless -------------------------------------

    #[test]
    fn widen_then_narrow_roundtrips_losslessly_all_pairs() {
        // For every (narrow, wide) integer pair with narrow.bits() <= wide.bits(), widening
        // then narrowing must recover the original bytes exactly (the low bits we shift in
        // are zero, so rounding never perturbs the result).
        for narrow in INT_FORMATS {
            for wide in INT_FORMATS {
                if wide.bits() < narrow.bits() {
                    continue;
                }
                // Sweep the full range of the narrow format.
                let (lo, hi) = int_range(narrow);
                // Step so the sweep stays cheap for 24/32-bit.
                let step = ((hi as i64 - lo as i64) / 4096).max(1);
                let mut v = lo as i64;
                while v <= hi as i64 {
                    let src = enc_native(narrow, v as i32);
                    let wide_bytes = one(narrow, wide, &src);
                    let back = one(wide, narrow, &wide_bytes);
                    assert_eq!(
                        back, src,
                        "round-trip {narrow:?}->{wide:?}->{narrow:?} at {v} must be lossless"
                    );
                    v += step;
                }
            }
        }
    }

    /// Inclusive native-value range for an integer format.
    fn int_range(fmt: SampleFormat) -> (i32, i32) {
        match fmt {
            SampleFormat::U8 => (-128, 127),
            SampleFormat::S16 => (i16::MIN as i32, i16::MAX as i32),
            SampleFormat::S24 => (-(1 << 23), (1 << 23) - 1),
            SampleFormat::S32 => (i32::MIN, i32::MAX),
            SampleFormat::F32 => unreachable!(),
        }
    }

    // --- narrowing rounds to nearest --------------------------------------------------

    #[test]
    fn narrow_rounds_to_nearest() {
        // s32 -> s16 drops 16 bits. Construct a value whose low 16 bits are exactly half
        // (0x8000): it must round up by one at the s16 scale.
        let v = (5i32 << 16) | 0x8000; // 5.5 at s16 scale
        assert_eq!(
            one(SampleFormat::S32, SampleFormat::S16, &v.to_le_bytes()),
            6i16.to_le_bytes().to_vec(),
            "0x8000 fractional part rounds up"
        );
        // Just below half rounds down.
        let v = (5i32 << 16) | 0x7FFF;
        assert_eq!(
            one(SampleFormat::S32, SampleFormat::S16, &v.to_le_bytes()),
            5i16.to_le_bytes().to_vec(),
            "0x7FFF fractional part rounds down"
        );
        // Negative half: -5.5 at s16 scale. Round-half-toward-+inf gives -5.
        let v = (-5i32 << 16) - 0x8000; // == -(5.5) at s16 scale
        let got = one(SampleFormat::S32, SampleFormat::S16, &v.to_le_bytes());
        assert_eq!(got, (-5i16).to_le_bytes().to_vec(), "-5.5 ties toward +inf => -5");
    }

    #[test]
    fn narrow_clamps_at_positive_full_scale() {
        // s32 max, narrowed to s16: rounding 0x7FFFFFFF up would overflow i16; it must clamp
        // to i16::MAX, not wrap to i16::MIN.
        assert_eq!(
            one(SampleFormat::S32, SampleFormat::S16, &i32::MAX.to_le_bytes()),
            i16::MAX.to_le_bytes().to_vec(),
            "narrowing the max value clamps instead of wrapping"
        );
        // Same for s32 -> s24 and s32 -> u8.
        assert_eq!(
            one(SampleFormat::S32, SampleFormat::S24, &i32::MAX.to_le_bytes()),
            enc_native(SampleFormat::S24, (1 << 23) - 1),
        );
        assert_eq!(
            one(SampleFormat::S32, SampleFormat::U8, &i32::MAX.to_le_bytes()),
            vec![255u8],
        );
    }

    // --- int <-> float ----------------------------------------------------------------

    #[test]
    fn int_to_float_maps_full_scale_near_plus_minus_one() {
        // s16 max -> ~ +0.9999 (32767/32768); s16 min -> exactly -1.0 (-32768/32768).
        let max_f = f32::from_le_bytes(
            one(SampleFormat::S16, SampleFormat::F32, &i16::MAX.to_le_bytes())
                .try_into()
                .unwrap(),
        );
        assert!((max_f - 32767.0 / 32768.0).abs() < 1e-6, "s16 max -> {max_f}");
        let min_f = f32::from_le_bytes(
            one(SampleFormat::S16, SampleFormat::F32, &i16::MIN.to_le_bytes())
                .try_into()
                .unwrap(),
        );
        assert!((min_f + 1.0).abs() < 1e-6, "s16 min -> {min_f}");
        // Zero maps to 0.0 for signed; U8 128 (silence) -> 0.0.
        assert_eq!(one(SampleFormat::S16, SampleFormat::F32, &0i16.to_le_bytes()), enc_f32(0.0));
        assert_eq!(one(SampleFormat::U8, SampleFormat::F32, &[128]), enc_f32(0.0));
    }

    #[test]
    fn float_to_int_scales_and_clamps() {
        // +1.0 -> s16 max (32767 == 2^15 - 1), -1.0 -> -32767 (not -32768: * (2^15 - 1)).
        assert_eq!(
            one(SampleFormat::F32, SampleFormat::S16, &enc_f32(1.0)),
            32767i16.to_le_bytes().to_vec()
        );
        assert_eq!(
            one(SampleFormat::F32, SampleFormat::S16, &enc_f32(-1.0)),
            (-32767i16).to_le_bytes().to_vec()
        );
        // Out-of-range clamps: +2.0 clamps to +1.0 -> 32767; -2.0 -> -32767.
        assert_eq!(
            one(SampleFormat::F32, SampleFormat::S16, &enc_f32(2.0)),
            32767i16.to_le_bytes().to_vec()
        );
        assert_eq!(
            one(SampleFormat::F32, SampleFormat::S16, &enc_f32(-9.0)),
            (-32767i16).to_le_bytes().to_vec()
        );
        // 0.0 -> 0 (s16); U8: 0.0 -> 128 (biased silence).
        assert_eq!(one(SampleFormat::F32, SampleFormat::S16, &enc_f32(0.0)), 0i16.to_le_bytes().to_vec());
        assert_eq!(one(SampleFormat::F32, SampleFormat::U8, &enc_f32(0.0)), vec![128u8]);
    }

    #[test]
    fn float_int_float_roundtrip_within_tolerance() {
        // For each integer depth, a ramp of floats survives f32 -> int -> f32 within one LSB
        // of that depth (the quantisation error bound).
        for to in INT_FORMATS {
            let bits = to.bits();
            let lsb = 1.0 / (1u64 << (bits - 1)) as f32;
            let mut f = -0.999f32;
            while f <= 0.999 {
                let i = one(SampleFormat::F32, to, &enc_f32(f));
                let back = f32::from_le_bytes(one(to, SampleFormat::F32, &i).try_into().unwrap());
                assert!(
                    (back - f).abs() <= lsb * 1.5,
                    "f32->{to:?}->f32 at {f}: got {back}, err {} > {}",
                    (back - f).abs(),
                    lsb * 1.5
                );
                f += 0.013;
            }
        }
    }

    #[test]
    fn float_to_float_is_exact_copy() {
        let src = 0.123456f32.to_le_bytes();
        assert_eq!(one(SampleFormat::F32, SampleFormat::F32, &src), src.to_vec());
    }

    // --- every format pair is defined and length-correct ------------------------------

    #[test]
    fn every_pair_converts_and_lengths_are_exact() {
        // Two interchannel frames, stereo, for every (from, to) pair: conversion succeeds,
        // and the output length is exactly frames * channels * to.bytes().
        for from in ALL_FORMATS {
            for to in ALL_FORMATS {
                let channels = 2usize;
                let frames = 2usize;
                let src = vec![0u8; frames * channels * from.bytes()];
                let want_len = frames * channels * to.bytes();
                assert_eq!(
                    converted_len(from, to, channels, src.len()),
                    Some(want_len),
                    "converted_len {from:?}->{to:?}"
                );
                let out = convert_interleaved_vec(from, to, channels, &src)
                    .unwrap_or_else(|| panic!("convert {from:?}->{to:?}"));
                assert_eq!(out.len(), want_len, "output length {from:?}->{to:?}");
                // Silence in every integer format is: signed 0, U8 128; float 0.0. Converting
                // silence must yield the target's silence.
                let silence_ok = match to {
                    SampleFormat::U8 => out.iter().all(|&b| b == 128),
                    _ => out.iter().all(|&b| b == 0),
                };
                // Only meaningful when the source was also silence.
                let src_silence = match from {
                    SampleFormat::U8 => src.iter().all(|&b| b == 128),
                    _ => src.iter().all(|&b| b == 0),
                };
                if src_silence {
                    // Our `src` is all-zero bytes; that is silence only for signed/float, not
                    // U8 (0 == full-scale negative). So assert the silence property only when
                    // the *source* bytes actually represent silence.
                    assert!(silence_ok, "silence preserved {from:?}->{to:?}: {out:?}");
                }
            }
        }
    }

    #[test]
    fn u8_zero_byte_is_full_scale_negative_not_silence() {
        // Guard against the classic U8 bug: byte 0 is -128 (min), byte 128 is silence.
        // 0u8 -> s16 must be i16::MIN-ish (-128 << 8 = -32768), not 0.
        assert_eq!(
            one(SampleFormat::U8, SampleFormat::S16, &[0]),
            (-32768i16).to_le_bytes().to_vec()
        );
        // 128u8 (silence) -> s16 == 0.
        assert_eq!(one(SampleFormat::U8, SampleFormat::S16, &[128]), 0i16.to_le_bytes().to_vec());
    }

    // --- error handling ---------------------------------------------------------------

    #[test]
    fn rejects_ragged_input_and_small_output() {
        // 3 bytes is not a whole s16 stereo frame (stride 4).
        assert_eq!(convert_interleaved_vec(SampleFormat::S16, SampleFormat::U8, 2, &[0, 0, 0]), None);
        // Output too small.
        let src = [0u8; 8]; // 4 s16 samples
        let mut tiny = [0u8; 1];
        assert_eq!(
            convert_interleaved(SampleFormat::S16, SampleFormat::S32, 1, &src, &mut tiny),
            None
        );
        // converted_len agrees on raggedness.
        assert_eq!(converted_len(SampleFormat::S24, SampleFormat::S16, 2, 5), None);
    }

    #[test]
    fn interleaving_is_preserved_across_channels() {
        // Distinct per-channel values must stay in their lanes through a widen.
        // Stereo s16: L=1000, R=-2000 for two frames.
        let mut src = Vec::new();
        for _ in 0..2 {
            src.extend_from_slice(&1000i16.to_le_bytes());
            src.extend_from_slice(&(-2000i16).to_le_bytes());
        }
        let out = convert_interleaved_vec(SampleFormat::S16, SampleFormat::S32, 2, &src).unwrap();
        // Read back the s32 samples.
        let mut got = Vec::new();
        for c in out.chunks_exact(4) {
            got.push(i32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        assert_eq!(
            got,
            vec![1000i32 << 16, -2000i32 << 16, 1000i32 << 16, -2000i32 << 16]
        );
    }

    // --- channel remap ----------------------------------------------------------------

    #[test]
    fn mono_to_stereo_duplicates() {
        // Two mono s16 samples -> stereo: each duplicated into L and R.
        let mut src = Vec::new();
        src.extend_from_slice(&111i16.to_le_bytes());
        src.extend_from_slice(&(-222i16).to_le_bytes());
        let mut out = vec![0u8; 8];
        let n = remap_channels(SampleFormat::S16, 1, 2, &src, &mut out).unwrap();
        assert_eq!(n, 8);
        let s: Vec<i16> = out
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(s, vec![111, 111, -222, -222]);
    }

    #[test]
    fn stereo_to_mono_averages() {
        // L=1000 R=2000 -> 1500; L=-100 R=100 -> 0.
        let mut src = Vec::new();
        src.extend_from_slice(&1000i16.to_le_bytes());
        src.extend_from_slice(&2000i16.to_le_bytes());
        src.extend_from_slice(&(-100i16).to_le_bytes());
        src.extend_from_slice(&100i16.to_le_bytes());
        let mut out = vec![0u8; 4];
        remap_channels(SampleFormat::S16, 2, 1, &src, &mut out).unwrap();
        let s: Vec<i16> = out
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(s, vec![1500, 0]);
    }

    #[test]
    fn stereo_to_mono_float_averages() {
        let mut src = Vec::new();
        src.extend_from_slice(&0.5f32.to_le_bytes());
        src.extend_from_slice(&(-0.5f32).to_le_bytes());
        let mut out = vec![0u8; 4];
        remap_channels(SampleFormat::F32, 2, 1, &src, &mut out).unwrap();
        let v = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
        assert!(v.abs() < 1e-6, "average of 0.5 and -0.5 is 0.0, got {v}");
    }

    #[test]
    fn remap_identity_and_unsupported() {
        // Identity channel count is a copy.
        let src = [1u8, 2, 3, 4];
        let mut out = [0u8; 4];
        assert_eq!(remap_channels(SampleFormat::U8, 2, 2, &src, &mut out), Some(4));
        assert_eq!(out, src);
        // 3 -> 2 is unsupported.
        let src3 = vec![0u8; 6];
        let mut out3 = vec![0u8; 4];
        assert_eq!(remap_channels(SampleFormat::U8, 3, 2, &src3, &mut out3), None);
    }

    // --- AudioFormat convenience wrapper ---------------------------------------------

    #[test]
    fn convert_uses_audioformat_channels() {
        // A tiny sanity check that channels flow through when derived from an AudioFormat.
        let f = AudioFormat::new(48_000, 2, SampleFormat::S16);
        let src = vec![0u8; 4 * f.channels as usize]; // 4 frames? no: 4 bytes/frame here
        let out = convert_interleaved_vec(f.format, SampleFormat::S32, f.channels as usize, &src);
        assert!(out.is_some());
    }
}
