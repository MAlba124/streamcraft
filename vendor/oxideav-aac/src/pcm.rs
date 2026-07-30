//! §4.6.11 time-domain output → integer PCM rendering.
//!
//! The §4.6.11 filterbank ([`crate::filterbank`]) emits one channel's
//! reconstructed time signal as `f64` samples already scaled to the
//! 16-bit full-scale amplitude domain (the `2/N` IMDCT normalisation
//! plus the §4.6.2.3.3 scalefactor gain land the dequantised, windowed,
//! overlap-added output directly on the `±32768` axis). This module
//! turns that floating-point time signal into the integer-PCM
//! representation a sink consumes, and interleaves a frame's channels.
//!
//! Two operations live here, both fully spec-determined:
//!
//! 1. **Rounding to the nearest integer.** ISO/IEC 14496-3 §1.3 defines
//!    the `NINT()` nearest-integer operator as *"Returns the nearest
//!    integer value to the real-valued argument. Half-integer values are
//!    rounded away from zero."* [`nint`] implements exactly that
//!    (`floor(x + 0.5)` for `x ≥ 0`, `ceil(x - 0.5)` for `x < 0`), which
//!    is the same tie-breaking rule the spec's `//` rounded-division and
//!    every other `NINT`-quoting clause use.
//!
//! 2. **Saturation to the output word.** A 16-bit signed sink represents
//!    `-32768 ..= 32767`; a sample whose magnitude overshoots that range
//!    (possible only on a clipped / full-scale input) saturates to the
//!    nearest representable extreme rather than wrapping. [`to_s16`]
//!    clamps after rounding.
//!
//! The conversion is *the only* output-rendering step the crate applies:
//! there is no resampler, no dither, and no channel remap. The integer
//! samples are produced in the filterbank's own time order; the optional
//! [`interleave_s16`] helper packs a frame's per-channel buffers into the
//! element-order interleaved layout a multi-channel sink expects.
//!
//! ## Provenance
//!
//! Every constant and rule here is from ISO/IEC 14496-3 (the §1.3
//! arithmetic-operator definitions and the §4.6.11 filterbank output
//! contract) staged under `docs/audio/aac/`. The full-scale `±32768`
//! amplitude domain is the filterbank's documented output scale (the
//! `2/N` IMDCT factor of [`crate::filterbank`]); this module adds only
//! the spec's `NINT()` rounding and the integer-word saturation.

use crate::{Error, Result};

/// The most negative value a 16-bit signed PCM word can hold.
pub const S16_MIN: i32 = -32768;
/// The most positive value a 16-bit signed PCM word can hold.
pub const S16_MAX: i32 = 32767;

/// ISO/IEC 14496-3 §1.3 `NINT()` — round a real value to the nearest
/// integer, with half-integers rounded **away from zero**.
///
/// `NINT(2.5) == 3`, `NINT(-2.5) == -3`, `NINT(2.4) == 2`,
/// `NINT(-2.4) == -2`. A non-finite input (`NaN` / `±∞`) has no nearest
/// integer; it returns `0.0` so a downstream cast cannot trap (the
/// filterbank never emits non-finite output for a well-formed stream,
/// but a hostile bitstream must not be able to poison the PCM cast).
#[must_use]
pub fn nint(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    if x >= 0.0 {
        (x + 0.5).floor()
    } else {
        (x - 0.5).ceil()
    }
}

/// Render one filterbank time-domain sample to a saturating 16-bit
/// signed PCM word.
///
/// Applies the §1.3 [`nint`] rounding then clamps to
/// [`S16_MIN`]`..=`[`S16_MAX`]. The clamp is a no-op for the
/// well-below-full-scale output of a dequantised LC stream; it only
/// engages on a clipped / full-scale signal whose rounded magnitude
/// would overflow the 16-bit word.
#[must_use]
pub fn to_s16(sample: f64) -> i16 {
    nint(sample).clamp(S16_MIN as f64, S16_MAX as f64) as i16
}

/// Render a whole channel's time signal to 16-bit PCM in place order,
/// returning a fresh `Vec<i16>` of the same length.
#[must_use]
pub fn channel_to_s16(samples: &[f64]) -> Vec<i16> {
    samples.iter().copied().map(to_s16).collect()
}

/// Interleave a frame's per-channel time signals into the element-order
/// interleaved 16-bit PCM layout a multi-channel sink consumes.
///
/// `channels[c][n]` is channel `c`'s sample `n`; the output is
/// `out[n * num_channels + c] = to_s16(channels[c][n])`. Every channel
/// buffer must be the same length (the §4.6.11 per-frame sample count,
/// `1024` for the 1024-line transform family); a length disagreement is
/// rejected with [`Error::PcmInvalid`]. An empty channel list yields an
/// empty buffer.
pub fn interleave_s16(channels: &[Vec<f64>]) -> Result<Vec<i16>> {
    if channels.is_empty() {
        return Ok(Vec::new());
    }
    let frame_len = channels[0].len();
    if channels.iter().any(|c| c.len() != frame_len) {
        return Err(Error::PcmInvalid);
    }
    let num_channels = channels.len();
    let mut out = Vec::with_capacity(frame_len * num_channels);
    for n in 0..frame_len {
        for ch in channels {
            out.push(to_s16(ch[n]));
        }
    }
    Ok(out)
}

/// Interleave a frame's per-channel time signals into a **caller-provided, reused** `Vec<i16>`,
/// in the same element-order interleaved layout as [`interleave_s16`].
///
/// `out` is cleared and refilled (`out[n·C + c] = to_s16(channels[c][n])`), so a caller that
/// parks one frame at a time reuses a single owned buffer across frames — it allocates once (when
/// the buffer first grows to the frame size) and thereafter only re-fills, imposing no
/// steady-state heap traffic (profluens patch — the AAC element parks its decoded frame here).
/// Generic over the per-channel allocator `CA` so it accepts arena-backed planar buffers
/// (`Vec<f64, &Arena>`) directly, without a heap copy of the planar data first.
///
/// Every channel buffer must be the same length; a disagreement is rejected with
/// [`Error::PcmInvalid`] (leaving `out` cleared). An empty channel list clears `out` and returns.
pub fn interleave_s16_into<CA: std::alloc::Allocator>(
    channels: &[Vec<f64, CA>],
    out: &mut Vec<i16>,
) -> Result<()> {
    out.clear();
    if channels.is_empty() {
        return Ok(());
    }
    let frame_len = channels[0].len();
    if channels.iter().any(|c| c.len() != frame_len) {
        return Err(Error::PcmInvalid);
    }
    let num_channels = channels.len();
    out.reserve(frame_len * num_channels);
    for n in 0..frame_len {
        for ch in channels {
            out.push(to_s16(ch[n]));
        }
    }
    Ok(())
}

/// Write a slice of already-rendered 16-bit PCM samples into a caller byte buffer as
/// little-endian bytes, returning the number of bytes written.
///
/// This is the second half of the [`interleave_s16_into`] → sink path: once a frame is parked as
/// interleaved `i16` (owned, reused), the emit step copies it into the pool slot here. `dst` must
/// hold at least `samples.len() · 2` bytes (checked; a short buffer is rejected with
/// [`Error::PcmInvalid`]).
pub fn s16_slice_le_into(samples: &[i16], dst: &mut [u8]) -> Result<usize> {
    let need = samples.len() * 2;
    if dst.len() < need {
        return Err(Error::PcmInvalid);
    }
    let mut o = 0;
    for &s in samples {
        let bytes = s.to_le_bytes();
        dst[o] = bytes[0];
        dst[o + 1] = bytes[1];
        o += 2;
    }
    Ok(need)
}

/// Interleave a frame's per-channel time signals **directly into a caller byte buffer** as
/// little-endian 16-bit PCM, returning the number of bytes written.
///
/// This is exactly [`interleave_s16`] followed by an `i16 → little-endian bytes` copy, fused
/// into one pass: `dst[(n·C + c)·2 ..][..2] = to_s16(channels[c][n]).to_le_bytes()`. It lets a
/// caller render straight into an output/pipeline buffer with no intermediate `Vec<i16>` and no
/// second copy (profluens patch — the AAC element interleaves into its pool slot here).
///
/// `dst` must hold at least `frame_len · num_channels · 2` bytes (checked; a short buffer is
/// rejected with [`Error::PcmInvalid`] rather than truncating). Every channel buffer must be the
/// same length, same as [`interleave_s16`]. An empty channel list writes nothing and returns 0.
pub fn interleave_s16_le_into(channels: &[Vec<f64>], dst: &mut [u8]) -> Result<usize> {
    if channels.is_empty() {
        return Ok(0);
    }
    let frame_len = channels[0].len();
    if channels.iter().any(|c| c.len() != frame_len) {
        return Err(Error::PcmInvalid);
    }
    let num_channels = channels.len();
    let need = frame_len * num_channels * 2;
    if dst.len() < need {
        return Err(Error::PcmInvalid);
    }
    let mut o = 0;
    for n in 0..frame_len {
        for ch in channels {
            let bytes = to_s16(ch[n]).to_le_bytes();
            dst[o] = bytes[0];
            dst[o + 1] = bytes[1];
            o += 2;
        }
    }
    Ok(need)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nint_rounds_half_away_from_zero() {
        // §1.3: half-integers round away from zero.
        assert_eq!(nint(2.5), 3.0);
        assert_eq!(nint(-2.5), -3.0);
        assert_eq!(nint(0.5), 1.0);
        assert_eq!(nint(-0.5), -1.0);
        assert_eq!(nint(1.5), 2.0);
        assert_eq!(nint(-1.5), -2.0);
    }

    #[test]
    fn nint_rounds_non_halves_to_nearest() {
        assert_eq!(nint(2.4), 2.0);
        assert_eq!(nint(2.6), 3.0);
        assert_eq!(nint(-2.4), -2.0);
        assert_eq!(nint(-2.6), -3.0);
        assert_eq!(nint(0.0), 0.0);
        assert_eq!(nint(-0.0), 0.0);
    }

    #[test]
    fn nint_non_finite_is_zero() {
        assert_eq!(nint(f64::NAN), 0.0);
        assert_eq!(nint(f64::INFINITY), 0.0);
        assert_eq!(nint(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn to_s16_saturates() {
        assert_eq!(to_s16(0.0), 0);
        assert_eq!(to_s16(100.4), 100);
        assert_eq!(to_s16(100.5), 101);
        assert_eq!(to_s16(-100.5), -101);
        // Beyond full scale clamps, not wraps.
        assert_eq!(to_s16(40000.0), S16_MAX as i16);
        assert_eq!(to_s16(-40000.0), S16_MIN as i16);
        // The exact extremes round-trip.
        assert_eq!(to_s16(32767.0), 32767);
        assert_eq!(to_s16(-32768.0), -32768);
        // 32767.5 rounds away from zero to 32768 then clamps to 32767.
        assert_eq!(to_s16(32767.5), 32767);
        // -32768.5 rounds to -32769 then clamps to -32768.
        assert_eq!(to_s16(-32768.5), -32768);
    }

    #[test]
    fn channel_to_s16_maps_each_sample() {
        let got = channel_to_s16(&[0.0, 1.4, 1.5, -1.5, 50000.0]);
        assert_eq!(got, vec![0, 1, 2, -2, S16_MAX as i16]);
    }

    #[test]
    fn interleave_two_channels() {
        let l = vec![0.0, 10.0, 20.0];
        let r = vec![1.0, 11.0, 21.0];
        let got = interleave_s16(&[l, r]).unwrap();
        assert_eq!(got, vec![0, 1, 10, 11, 20, 21]);
    }

    #[test]
    fn interleave_single_channel_is_identity_order() {
        let mono = vec![3.4, 3.5, -3.5];
        let got = interleave_s16(&[mono]).unwrap();
        assert_eq!(got, vec![3, 4, -4]);
    }

    #[test]
    fn interleave_empty_is_empty() {
        assert!(interleave_s16(&[]).unwrap().is_empty());
    }

    #[test]
    fn interleave_rejects_length_mismatch() {
        let l = vec![0.0, 1.0];
        let r = vec![0.0, 1.0, 2.0];
        assert!(matches!(interleave_s16(&[l, r]), Err(Error::PcmInvalid)));
    }

    #[test]
    fn interleave_le_into_matches_the_two_step_path() {
        // The fused writer must be byte-for-byte identical to `interleave_s16` followed by a
        // little-endian byte copy, over saturating and rounding cases and multiple channels.
        let a = vec![0.0, 10.4, 10.5, -10.5, 40000.0, -40000.0, 32767.5, -32768.5];
        let b = vec![1.0, -1.0, 100.0, -100.0, 0.5, -0.5, 12345.0, -12345.0];
        let c = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        let channels = [a, b, c];

        let reference: Vec<u8> = interleave_s16(&channels)
            .unwrap()
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();

        let mut dst = vec![0u8; reference.len()];
        let n = interleave_s16_le_into(&channels, &mut dst).unwrap();
        assert_eq!(n, reference.len());
        assert_eq!(dst, reference);
    }

    #[test]
    fn interleave_le_into_rejects_short_buffer() {
        let l = vec![0.0, 1.0];
        let r = vec![2.0, 3.0];
        let mut dst = [0u8; 7]; // need 2*2*2 = 8
        assert!(matches!(
            interleave_s16_le_into(&[l, r], &mut dst),
            Err(Error::PcmInvalid)
        ));
    }

    #[test]
    fn interleave_le_into_empty_writes_nothing() {
        let mut dst = [0u8; 4];
        assert_eq!(interleave_s16_le_into(&[], &mut dst).unwrap(), 0);
    }

    #[test]
    fn interleave_into_matches_interleave_s16_and_reuses() {
        // The reused-buffer interleave must produce the same samples as `interleave_s16`, and
        // refilling clears the prior contents (so a shorter second frame does not keep stale
        // tail samples).
        let l = vec![0.0, 10.0, 20.0];
        let r = vec![1.0, 11.0, 21.0];
        let want = interleave_s16(&[l.clone(), r.clone()]).unwrap();
        let mut buf: Vec<i16> = vec![99; 100]; // pre-seeded to prove the clear
        interleave_s16_into(&[l, r], &mut buf).unwrap();
        assert_eq!(buf, want);
        // Reuse for a shorter frame — no stale samples survive.
        let mono = vec![3.4, 3.5];
        interleave_s16_into(&[mono], &mut buf).unwrap();
        assert_eq!(buf, vec![3, 4]);
    }

    #[test]
    fn interleave_into_rejects_length_mismatch_and_clears() {
        let l = vec![0.0, 1.0];
        let r = vec![0.0, 1.0, 2.0];
        let mut buf: Vec<i16> = vec![7, 7];
        assert!(matches!(interleave_s16_into(&[l, r], &mut buf), Err(Error::PcmInvalid)));
        assert!(buf.is_empty(), "a rejected interleave leaves the buffer cleared");
    }

    #[test]
    fn s16_slice_le_writes_little_endian() {
        let samples = [0i16, 1, -1, 258, i16::MIN, i16::MAX];
        let mut dst = vec![0u8; samples.len() * 2];
        let n = s16_slice_le_into(&samples, &mut dst).unwrap();
        assert_eq!(n, dst.len());
        let want: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        assert_eq!(dst, want);
    }

    #[test]
    fn s16_slice_le_rejects_short_buffer() {
        let samples = [1i16, 2, 3];
        let mut dst = [0u8; 5]; // need 6
        assert!(matches!(s16_slice_le_into(&samples, &mut dst), Err(Error::PcmInvalid)));
    }
}
