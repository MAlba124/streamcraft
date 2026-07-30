//! CELT coarse-energy Laplace symbol decode (`ec_laplace_decode`).
//!
//! RFC 6716 §4.3.2.1 codes the coarse per-band energy prediction error
//! with a Laplace-like distribution whose `{probability, decay}`
//! parameters live in the `e_prob_model` table (see
//! [`crate::celt_e_prob_model`]). The RFC names the symbol decoder
//! `ec_laplace_decode()` but defers the recurrence itself to its
//! reference source; this module recovers that recurrence as the
//! wire-format range-coder interval narrowing a conforming decoder MUST
//! perform to stay in lockstep with the encoder
//! (`docs/audio/celt/spec/celt-laplace-decode.md`).
//!
//! The decoder draws a 15-bit cumulative position from the range coder
//! (`ec_decode_bin(15)`), walks the decaying body magnitude by
//! magnitude until the per-magnitude mass reaches the floor
//! [`LAPLACE_MINP`], covers the flat tail with one division, resolves
//! the sign, and reports the consumed sub-interval back to the range
//! coder via `ec_dec_update(fl, min(fl+fs, 32768), 32768)`.
//!
//! # Scope
//!
//! This module is the **15-bit Laplace path** only — the path the
//! coarse-energy decoder takes when at least 15 range-coder bits of
//! budget remain for the band. The 2-bit / 1-bit / 0-bit fallbacks
//! (RFC 6716 §4.3.2.1 budget accounting) are handled by the
//! coarse-energy driver, not here.

/// `log2` of the floor probability (`LAPLACE_LOG_MINP`).
///
/// The right shift in the flat tail is `>> (LAPLACE_LOG_MINP + 1)`,
/// i.e. division by `2 * LAPLACE_MINP`.
pub const LAPLACE_LOG_MINP: u32 = 0;

/// Minimum probability of any energy delta, out of 32768 (one
/// range-coder tick): `1 << LAPLACE_LOG_MINP`.
pub const LAPLACE_MINP: u32 = 1 << LAPLACE_LOG_MINP;

/// Number of magnitudes (in one direction) guaranteed a probability
/// above the floor before the flat tail kicks in (`LAPLACE_NMIN`).
pub const LAPLACE_NMIN: u32 = 16;

/// The range-coder total the Laplace path works against: a 15-bit total
/// (`ec_decode_bin(dec, 15)` draws into `[0, 32768)`).
pub const LAPLACE_TOTAL: u32 = 1 << 15;

/// The Q14 unit of the geometric `decay` ratio (out of 16384).
pub const LAPLACE_DECAY_UNIT: u32 = 1 << 14;

/// Promote an `e_prob_model` `prob` byte (Q8) to the value-0
/// sub-interval size `fs` (Q15) the Laplace decoder expects: `prob << 7`.
#[inline]
#[must_use]
pub const fn prob_to_fs(prob: u8) -> u32 {
    (prob as u32) << 7
}

/// Promote an `e_prob_model` `decay` byte (Q8) to the geometric ratio
/// `decay` (Q14) the Laplace decoder expects: `decay << 6`.
#[inline]
#[must_use]
pub const fn decay_byte_to_q14(decay: u8) -> u32 {
    (decay as u32) << 6
}

/// The "first frequency" helper: the size of the **first off-centre**
/// magnitude's sub-interval, before the floor is added.
///
/// `fs` is the value-0 sub-interval size (Q15); `decay` is the
/// geometric ratio (Q14). Returns the magnitude-1 probability mass
/// (before `LAPLACE_MINP` is added):
///
/// ```text
///   ft   = 32768 - LAPLACE_MINP * (2*LAPLACE_NMIN) - fs
///   freq = ft * (16384 - decay) >> 15
/// ```
#[inline]
#[must_use]
fn laplace_get_freq1(fs: u32, decay: u32) -> u32 {
    // ft is the mass available to the decaying body: the full total
    // minus the floor reserved for the 2*LAPLACE_NMIN guaranteed tail
    // entries, minus the value-0 mass. Saturating to keep the helper
    // total even if a degenerate fs/decay pair would overshoot.
    let ft = LAPLACE_TOTAL
        .saturating_sub(LAPLACE_MINP * (2 * LAPLACE_NMIN))
        .saturating_sub(fs);
    // Scale by the Q14 complement (16384 - decay), halved by >>15:
    // (u64 product keeps the intermediate exact; the operands are well
    // under 2^32 so a u64 multiply cannot overflow).
    let prod = (ft as u64) * ((LAPLACE_DECAY_UNIT - decay) as u64);
    (prod >> 15) as u32
}

/// Decode one signed coarse-energy symbol `qi` from the range decoder
/// via the §4.3.2.1 Laplace recurrence.
///
/// `fs` is the value-0 sub-interval size (Q15, from
/// [`prob_to_fs`]); `decay` is the geometric ratio (Q14, from
/// [`decay_byte_to_q14`]). The recurrence draws a 15-bit cumulative
/// position, narrows the range coder to the chosen symbol's sub-interval
/// `[fl, min(fl+fs, 32768))`, and returns the signed magnitude.
///
/// The range decoder's sticky error flag (consulted via
/// [`crate::range_decoder::RangeDecoder::has_error`]) is set by the
/// underlying `ec_dec_update` if the reported interval is degenerate;
/// the caller should check it after a run of decodes.
pub fn ec_laplace_decode(
    rd: &mut crate::range_decoder::RangeDecoder<'_>,
    mut fs: u32,
    decay: u32,
) -> i32 {
    // Step 1 — draw the cumulative position in [0, 32768).
    let fm = rd.decode_bin(15);
    let mut val: i32 = 0;
    let mut fl: u32 = 0;

    // Step 2 — zero symbol? If fm < fs the magnitude is 0; skip the body.
    if fm >= fs {
        // Step 3 — step onto magnitude 1.
        val = 1;
        fl = fs;
        fs = laplace_get_freq1(fs, decay) + LAPLACE_MINP;

        // Step 4 — walk down the decaying body. The boundary test
        // `fm >= fl + 2*fs` compares against twice the current mass
        // because the symmetric +/- pair jointly occupies 2*fs.
        while fs > LAPLACE_MINP && fm >= fl + 2 * fs {
            fs *= 2;
            fl += fs;
            // Geometric shrink of the body mass, removing the two floor
            // ticks first, then re-adding one floor tick.
            fs = ((fs - 2 * LAPLACE_MINP) * decay) >> 15;
            fs += LAPLACE_MINP;
            val += 1;
        }

        // Step 5 — flat tail. Every remaining magnitude has only the
        // floor probability; cover the remaining distance in one
        // division by 2*LAPLACE_MINP.
        if fs <= LAPLACE_MINP {
            let di = (fm - fl) >> (LAPLACE_LOG_MINP + 1);
            val += di as i32;
            fl += 2 * di * LAPLACE_MINP;
        }

        // Step 6 — resolve the sign and final interval. The chosen
        // magnitude occupies [fl, fl+fs), split into a negative lower
        // half and a positive upper half.
        if fm < fl + fs {
            val = -val;
        } else {
            fl += fs;
        }
    }

    // Step 7 — report consumption to the range coder, clamping the high
    // end to the 32768 total.
    let fh = (fl + fs).min(LAPLACE_TOTAL);
    rd.ec_dec_update(fl, fh, LAPLACE_TOTAL);
    val
}

/// Encode one signed coarse-energy symbol via the §4.3.2.1 Laplace
/// model — the exact write-side mirror of [`ec_laplace_decode`],
/// transcribed from the reference listing's encoder half.
///
/// `value` is clamped in place when the coded alphabet cannot represent
/// it (the flat tail runs out of probability floor ticks near ±32768);
/// the caller must use the (possibly clamped) value for its own
/// prediction feedback, exactly as the decoder will reconstruct it.
pub fn ec_laplace_encode(
    enc: &mut crate::range_encoder::RangeEncoder,
    value: &mut i32,
    mut fs: u32,
    decay: u32,
) {
    let mut fl: u32 = 0;
    let mut val = *value;
    if val != 0 {
        // s = -(val < 0); val = |val| via the two's-complement fold.
        let s: i32 = if val < 0 { -1 } else { 0 };
        val = (val + s) ^ s;
        fl = fs;
        fs = laplace_get_freq1(fs, decay);
        // Search the decaying part of the PDF.
        let mut i: i32 = 1;
        while fs > 0 && i < val {
            fs *= 2;
            fl += fs + 2 * LAPLACE_MINP;
            fs = (fs * decay) >> 15;
            i += 1;
        }
        if fs == 0 {
            // Everything beyond this point has probability LAPLACE_MINP.
            let mut ndi_max: i32 =
                ((LAPLACE_TOTAL - fl + LAPLACE_MINP - 1) >> LAPLACE_LOG_MINP) as i32;
            ndi_max = (ndi_max - s) >> 1;
            let di: i32 = (val - i).min(ndi_max - 1);
            fl += ((2 * di + 1 + s) as u32) * LAPLACE_MINP;
            fs = LAPLACE_MINP.min(LAPLACE_TOTAL - fl);
            *value = (i + di + s) ^ s;
        } else {
            fs += LAPLACE_MINP;
            if s == 0 {
                // fl += fs & ~s (s == 0 → add; s == -1 → skip).
                fl += fs;
            }
        }
        debug_assert!(fl + fs <= LAPLACE_TOTAL);
        debug_assert!(fs > 0);
    }
    enc.encode_bin(fl, fl + fs, 15);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;

    #[test]
    fn laplace_encode_decode_roundtrips_over_the_e_prob_surface() {
        // Every (LM, intra, band) Laplace parameter pair from the
        // §4.3.2.1 tables, over the full useful qi range: the encoder's
        // (possibly clamped) value must decode back exactly.
        use crate::celt_e_prob_model::{e_prob_pair, EnergyPredictionMode};
        use crate::range_encoder::RangeEncoder;
        for lm in 0..4u32 {
            for mode in [EnergyPredictionMode::Inter, EnergyPredictionMode::Intra] {
                for band in [0u32, 3, 10, 20] {
                    let pair = e_prob_pair(lm, mode, band).unwrap();
                    let fs = u32::from(pair.prob) << 7;
                    let decay = u32::from(pair.decay) << 6;
                    let mut enc = RangeEncoder::new();
                    let mut expect = Vec::new();
                    for q in -40i32..=40 {
                        let mut v = q;
                        ec_laplace_encode(&mut enc, &mut v, fs, decay);
                        expect.push(v);
                    }
                    let buf = enc.finish();
                    let mut rd = RangeDecoder::new(&buf);
                    for (idx, &want) in expect.iter().enumerate() {
                        let got = ec_laplace_decode(&mut rd, fs, decay);
                        assert_eq!(got, want, "lm={lm} band={band} sym#{idx}");
                    }
                    assert!(!rd.has_error());
                }
            }
        }
    }

    #[test]
    fn constants_match_table() {
        assert_eq!(LAPLACE_LOG_MINP, 0);
        assert_eq!(LAPLACE_MINP, 1);
        assert_eq!(LAPLACE_NMIN, 16);
        assert_eq!(LAPLACE_TOTAL, 32768);
        assert_eq!(LAPLACE_DECAY_UNIT, 16384);
    }

    #[test]
    fn scaling_shifts() {
        // prob byte (Q8) << 7 -> fs (Q15); decay byte (Q8) << 6 -> Q14.
        assert_eq!(prob_to_fs(72), 72 << 7);
        assert_eq!(prob_to_fs(72), 9216);
        assert_eq!(decay_byte_to_q14(127), 127 << 6);
        assert_eq!(decay_byte_to_q14(127), 8128);
    }

    #[test]
    fn get_freq1_worked_example() {
        // From celt-laplace-decode.md §6: fs0 = 72<<7 = 9216, decay =
        // 127<<6 = 8128. ft = 32768 - 2*16 - 9216 = 23520;
        // freq = 23520 * (16384 - 8128) >> 15.
        // Evaluating the stated formula exactly: 23520 * 8256 =
        // 194_181_120, and 194_181_120 >> 15 = 5925. (The doc's §6
        // prose shows the intermediate as 194213120 → 5927, a
        // transcription slip in the worked example; the formula itself
        // — which this helper implements verbatim — yields 5925.)
        let fs0 = prob_to_fs(72);
        let decay = decay_byte_to_q14(127);
        assert_eq!(23520u64 * 8256, 194_181_120);
        assert_eq!(laplace_get_freq1(fs0, decay), 5925);
    }

    #[test]
    fn body_step_decay_ratio() {
        // §6 sanity: with decay = 6000 the mass shrinks by ~decay/16384.
        // Reproduce the single-step arithmetic the loop performs.
        let decay = 6000u32;
        let fs = 5000u32;
        let doubled = fs * 2; // 10000
        let shrunk = ((doubled - 2 * LAPLACE_MINP) * decay) >> 15;
        assert_eq!(shrunk, 1830);
        assert_eq!(shrunk + LAPLACE_MINP, 1831);
    }

    /// Decode is the inverse of a hand-built interval: when fm lands in
    /// the value-0 sub-interval [0, fs) the symbol must be 0, and the
    /// range coder must be left consistent (no error latched).
    #[test]
    fn zero_symbol_for_central_draw() {
        // A buffer whose leading bits keep the first 15-bit draw small.
        let buf = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut rd = RangeDecoder::new(&buf);
        let qi = ec_laplace_decode(&mut rd, prob_to_fs(72), decay_byte_to_q14(127));
        assert_eq!(qi, 0);
        assert!(!rd.has_error());
    }

    /// A large value-0 mass means almost every draw decodes to 0; a tiny
    /// value-0 mass means most draws decode non-zero. Exercise both so
    /// the body/tail branches run and never latch an error.
    #[test]
    fn body_and_tail_branches_run_clean() {
        for byte in [0xffu8, 0x80, 0x40, 0x01] {
            let buf = [byte, byte, byte, byte, byte, byte, byte, byte];
            let mut rd = RangeDecoder::new(&buf);
            // Tiny fs (prob=1) forces non-zero magnitudes; small decay
            // makes the body decay fast so the tail branch is reached.
            let _ = ec_laplace_decode(&mut rd, prob_to_fs(1), decay_byte_to_q14(8));
            assert!(!rd.has_error(), "byte {byte:#x} latched an error");
        }
    }
}
