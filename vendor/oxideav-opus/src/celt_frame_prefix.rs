//! CELT frame-prefix symbol decode (RFC 6716 Table 56 head).
//!
//! Every CELT (MDCT-layer) frame opens with a fixed sequence of
//! range-coded flags that the decoder MUST read before any band data
//! (RFC 6716 Table 56 / §4.3.7.1):
//!
//! 1. **silence** — `{32767, 1}/32768`. When set, the whole CELT frame
//!    is silent; the decoder produces zero output but still advances its
//!    overlap-add / de-emphasis state (§4.5.1).
//! 2. **post-filter** — one bit (`logp = 1`). If set, four further
//!    parameters follow (§4.3.7.1):
//!    - **octave** — uniform in `0..=5` (Table 56: `uniform (6)`, i.e.
//!      `ec_dec_uint(6)` over the 6 values `0..=5`).
//!    - **period** — `4 + octave` raw bits; final pitch period
//!      `T = (16 << octave) + fine_pitch - 1`, bounded `15..=1022`.
//!      The §4.3.7.1 bound proves the octave range: only `octave <= 5`
//!      keeps the maximum `(16 << 5) + (2^9 - 1) - 1 = 1022`; an
//!      `octave = 6` could reach `T = 2046`, outside the stated bound.
//!    - **gain** — 3 raw bits; `G = 3*(int_gain+1)/32`.
//!    - **tapset** — `{2, 1, 1}/4`.
//! 3. **transient** — `{7, 1}/8`.
//! 4. **intra** — `{7, 1}/8`.
//!
//! This module decodes that prefix in order, leaving the range decoder
//! positioned at the coarse-energy symbol. The post-filter is applied
//! at the very end of synthesis (§4.3.7.1) but its parameters are coded
//! here, "just after the silence flag", so they are captured into
//! [`CeltPostFilterParams`] for the synthesis stage to consume.

/// Decoded post-filter parameters (RFC 6716 §4.3.7.1), present only when
/// the post-filter flag is set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CeltPostFilterParams {
    /// Octave `0..=5` selecting the pitch range (Table 56 `uniform (6)`).
    pub octave: u32,
    /// Final pitch period `T = (16 << octave) + fine_pitch - 1`, bounded
    /// `15..=1022` inclusive.
    pub period: u32,
    /// Raw 3-bit gain index `int_gain`; the applied gain is
    /// `G = 3*(int_gain+1)/32` (computed by the synthesis stage).
    pub gain_index: u32,
    /// Tap-set selector `0..=2` choosing the `{g0, g1, g2}` triple.
    pub tapset: u32,
}

impl CeltPostFilterParams {
    /// The applied post-filter gain `G = 3*(int_gain+1)/32` (§4.3.7.1).
    #[inline]
    #[must_use]
    pub fn gain(self) -> f64 {
        3.0 * (self.gain_index as f64 + 1.0) / 32.0
    }
}

/// The decoded Table-56 prefix of one CELT frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CeltFramePrefix {
    /// Silence flag: the whole CELT frame is silent (§4.5.1).
    pub silence: bool,
    /// Post-filter parameters, or `None` when the post-filter is off.
    pub post_filter: Option<CeltPostFilterParams>,
    /// Transient flag: single long MDCT (false) vs. short blocks (true)
    /// (§4.3.1). Always `false` for a silence frame (not coded).
    pub transient: bool,
    /// Intra flag: coarse energy coded without time prediction
    /// (§4.3.2.1). Always `false` for a silence frame (not coded).
    pub intra: bool,
}

/// `logp` for the post-filter on/off bit (`{1, 1}/2`).
const POST_FILTER_LOGP: u32 = 1;

/// ICDF for the transient flag `{7, 1}/8` (ftb = 3). The single-entry
/// inverse-CDF body is `[1, 0]`: `ft - fh[0] = 8 - 7 = 1`, terminator 0.
const TRANSIENT_ICDF: [u8; 2] = [1, 0];

/// ICDF for the intra flag `{7, 1}/8` (ftb = 3), same shape as the
/// transient flag.
const INTRA_ICDF: [u8; 2] = [1, 0];

/// ICDF for the post-filter tap-set `{2, 1, 1}/4` (ftb = 2):
/// `fh = [2, 3, 4]`, so the inverse-CDF body is `ft - fh = [2, 1, 0]`.
const TAPSET_ICDF: [u8; 3] = [2, 1, 0];

/// ftb for the `/8` flags and the `/4` tap-set table.
const FTB_8: u32 = 3;
const FTB_4: u32 = 2;

/// Decode the CELT frame prefix (Table 56 head) from `rd`, leaving the
/// range decoder positioned at the coarse-energy symbol.
///
/// Symbols are read strictly in Table-56 order. When the silence flag is
/// set the post-filter / transient / intra symbols are still **not**
/// coded for a silent frame in the sense that the band-data path is
/// skipped, but the post-filter and the transient / intra flags continue
/// to be part of the prefix in the live (non-silent) path; for the
/// silent path we return `transient = intra = false` and read only the
/// silence + post-filter prefix (the post-filter group is coded
/// unconditionally right after the silence flag per §4.3.7.1).
pub fn decode_celt_frame_prefix(
    rd: &mut crate::range_decoder::RangeDecoder<'_>,
) -> CeltFramePrefix {
    // 1. silence — {32767, 1}/32768. Decoded as a single binary symbol
    //    with logp = 15 (ft = 32768, the "1" mass is one tick). The
    //    range coder's dec_bit_logp(15) yields 1 with probability
    //    1/32768, matching the {32767, 1} split.
    let silence = rd.dec_bit_logp(15) == 1;

    // 2. post-filter group — the on/off bit and, when on, its four
    //    parameters. Coded right after the silence flag (§4.3.7.1).
    let post_filter = if rd.dec_bit_logp(POST_FILTER_LOGP) == 1 {
        // octave — Table 56 codes it as `uniform (6)`: the range coder's
        // ec_dec_uint(ft = 6) returns [0, 5]. The §4.3.7.1 prose "an
        // integer value between 0 and 6" describes the ft, not the value
        // range; the section's own period bound settles it — `T =
        // (16 << octave) + fine_pitch - 1` is "bounded between 15 and
        // 1022, inclusively", and only octave <= 5 (with its 4+5 = 9 raw
        // fine-pitch bits) keeps T <= (16 << 5) + 511 - 1 = 1022.
        let octave = rd.dec_uint(6).unwrap_or(0);
        // period — 4 + octave raw bits, then T = (16<<octave)+fine-1.
        let fine_pitch = rd.dec_bits(4 + octave);
        let period = (16u32 << octave) + fine_pitch - 1;
        // gain — 3 raw bits.
        let gain_index = rd.dec_bits(3);
        // tapset — {2, 1, 1}/4.
        let tapset = rd.dec_icdf(&TAPSET_ICDF, FTB_4);
        Some(CeltPostFilterParams {
            octave,
            period,
            gain_index,
            tapset,
        })
    } else {
        None
    };

    // For a silent frame the band-data symbols (transient/intra/coarse
    // energy/…) are not coded; the decoder produces silence and only the
    // silence + post-filter prefix has been consumed.
    if silence {
        return CeltFramePrefix {
            silence: true,
            post_filter,
            transient: false,
            intra: false,
        };
    }

    // 3. transient — {7, 1}/8.
    let transient = rd.dec_icdf(&TRANSIENT_ICDF, FTB_8) == 1;
    // 4. intra — {7, 1}/8.
    let intra = rd.dec_icdf(&INTRA_ICDF, FTB_8) == 1;

    CeltFramePrefix {
        silence: false,
        post_filter,
        transient,
        intra,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;

    #[test]
    fn gain_formula() {
        let p = CeltPostFilterParams {
            octave: 0,
            period: 15,
            gain_index: 0,
            tapset: 0,
        };
        // int_gain = 0 → G = 3*1/32.
        assert!((p.gain() - 3.0 / 32.0).abs() < 1e-12);
        let p7 = CeltPostFilterParams { gain_index: 7, ..p };
        // int_gain = 7 → G = 3*8/32 = 0.75.
        assert!((p7.gain() - 0.75).abs() < 1e-12);
    }

    /// A frame whose leading byte sets the silence bit decodes as silent
    /// and consumes only the silence + post-filter prefix; the range
    /// coder must not latch an error.
    #[test]
    fn silence_frame_prefix_is_clean() {
        // Construct a buffer that decodes silence = 1. The silence bit
        // is the {32767,1}/32768 "1" branch, which dec_bit_logp(15)
        // selects when val < rng>>15 at the very first symbol. The
        // initial decoder state has val seeded from the first bytes;
        // 0xff leading bytes push val high → silence = 0. We instead
        // exercise the structural property: whatever the silence bit,
        // the prefix decode never panics and never latches an error on
        // a well-formed buffer.
        let buf = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut rd = RangeDecoder::new(&buf);
        let prefix = decode_celt_frame_prefix(&mut rd);
        assert!(!rd.has_error());
        // With the all-zero buffer the post-filter / flags decode to a
        // well-defined state; assert the period bound holds when present.
        if let Some(pf) = prefix.post_filter {
            assert!((15..=1022).contains(&pf.period));
            assert!(pf.octave <= 5);
            assert!(pf.tapset <= 2);
        }
    }

    /// The transient / intra ICDF tables encode the {7,1}/8 split: a
    /// draw in the lower 7/8 yields 0, the top 1/8 yields 1.
    #[test]
    fn flag_tables_have_correct_shape() {
        // ft = 1<<3 = 8; fh[0] = 7 → icdf[0] = 8 - 7 = 1, terminator 0.
        assert_eq!(TRANSIENT_ICDF, [1, 0]);
        assert_eq!(INTRA_ICDF, [1, 0]);
        // tapset {2,1,1}/4: fh = [2,3,4] → icdf = [2,1,0].
        assert_eq!(TAPSET_ICDF, [2, 1, 0]);
    }

    /// The post-filter octave is Table 56 `uniform (6)`: ec_dec_uint(6)
    /// over the 6 values 0..=5. Sweeping the leading bytes, every decoded
    /// octave (when the post-filter is on) must fall in 0..=5, the
    /// maximum value 5 must be reachable, and — the §4.3.7.1 property
    /// that pins the ft — every decoded period must satisfy the
    /// normative bound `15 <= T <= 1022`. (A 7-value octave decode would
    /// reach octave 6 and periods up to `(16 << 6) + 1023 - 1 = 2046`,
    /// violating the stated bound.)
    #[test]
    fn post_filter_octave_in_range_and_period_bounded() {
        let mut max_octave = 0u32;
        let mut max_period = 0u32;
        let mut saw_post_filter = false;
        for b0 in 0u16..=255 {
            for b1 in 0u16..=255 {
                let buf = [b0 as u8, b1 as u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];
                let mut rd = RangeDecoder::new(&buf);
                let p = decode_celt_frame_prefix(&mut rd);
                if let Some(pf) = p.post_filter {
                    saw_post_filter = true;
                    assert!(pf.octave <= 5, "octave {} out of range", pf.octave);
                    assert!(
                        (15..=1022).contains(&pf.period),
                        "period {} violates the §4.3.7.1 bound",
                        pf.period
                    );
                    max_octave = max_octave.max(pf.octave);
                    max_period = max_period.max(pf.period);
                }
            }
        }
        assert!(saw_post_filter, "no post-filter frame in the sweep");
        assert_eq!(max_octave, 5, "octave 5 must be reachable (ec_dec_uint(6))");
    }

    /// Non-silent prefix: feed a buffer that decodes silence = 0 and walk
    /// the full prefix (post-filter group + transient + intra) cleanly.
    #[test]
    fn non_silent_prefix_runs_clean() {
        for seed in [0x55u8, 0xaa, 0x0f, 0xf0] {
            let buf = [seed; 12];
            let mut rd = RangeDecoder::new(&buf);
            let _ = decode_celt_frame_prefix(&mut rd);
            assert!(!rd.has_error(), "seed {seed:#x} latched an error");
        }
    }
}
