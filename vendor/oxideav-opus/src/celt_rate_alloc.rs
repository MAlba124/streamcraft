//! CELT §4.3.3 bit allocation — the decode-side interpolated
//! bits-to-pulses computation (RFC 6716 §4.3.3, pp. 112–116).
//!
//! Given the explicitly-signalled allocation header (per-band boosts,
//! trim, and the reservation gates), the §4.3.3 *implicit* allocation
//! deterministically splits the remaining frame budget into a per-band
//! PVQ budget (in 1/8-bit units), a per-band fine-energy bit count, and
//! a per-band fine-energy priority flag, while also decoding the three
//! symbols the bitstream interleaves *into* the computation: the
//! band-skip flags, the intensity-stereo band index, and the
//! dual-stereo flag.
//!
//! RFC 6716 §1 and §6 give the Appendix A reference decoder precedence
//! over the prose narrative; this module is a faithful port of the
//! allocation procedure as the normative reference decoder specifies it
//! (`compute_allocation` / `interp_bits2pulses`, rate.c, and the
//! `bits2pulses` / `pulses2bits` / `get_pulses` cache accessors,
//! rate.h), operating on:
//!
//! * the Table 57 static allocation matrix
//!   ([`crate::celt_static_alloc::STATIC_ALLOC`]),
//! * the §4.3.3 per-band caps ([`crate::celt_cache_caps50`]),
//! * the §4.3.4.1 pulse-cost cache ([`crate::celt_pulse_cache`],
//!   LM-major mapping),
//! * the `LOG2_FRAC_TABLE` intensity reservation costs
//!   ([`crate::celt_log2_frac_table::LOG2_FRAC_TABLE`]), and
//! * the per-band `logN` table ([`LOG_N_400`], the Q3 `log2` of each
//!   band's short-MDCT width, from `docs/audio/opus/tables/log-n400.csv`).
//!
//! All arithmetic is exact integer arithmetic in `BITRES = 3` (1/8-bit)
//! units; there are no floating-point decisions anywhere in the
//! allocator, so this port is bit-exact by construction.
//!
//! ## Provenance
//!
//! RFC 6716 §4.3.3 narrative + the normative Appendix A reference
//! decoder (both from `docs/audio/opus/rfc6716-opus.txt`; the Appendix A
//! reference implementation is part of the staged RFC document and, per
//! §6, takes precedence over the prose). No external library source was
//! consulted.

use crate::celt_band_layout::{celt_band_bins_per_channel, CeltFrameSize, CELT_NUM_BANDS};
use crate::celt_log2_frac_table::LOG2_FRAC_TABLE;
use crate::celt_pulse_cache::{CACHE_BITS50, CACHE_INDEX50};
use crate::celt_static_alloc::STATIC_ALLOC;
use crate::range_decoder::RangeDecoder;

/// `BITRES`: allocation arithmetic runs in 1/8-bit units (Q3).
pub const BITRES: u32 = 3;

/// Maximum fine-energy bits per band per channel (§4.3.2.2).
pub const MAX_FINE_BITS: i32 = 8;

/// The fine-energy allocation offset (1/8 bits) subtracted from a
/// band's "fair share" when splitting its budget between fine energy
/// and PVQ.
const FINE_OFFSET: i32 = 21;

/// Number of bisection refinement steps for the interpolated
/// allocation.
const ALLOC_STEPS: u32 = 6;

/// Number of rows in the Table 57 static allocation matrix.
const N_ALLOC_VECTORS: i32 = 11;

/// `logN400`: Q3 `log2` of each band's short-MDCT width (the
/// `log2_frac(width, 3)` of the Table 55 widths). From
/// `docs/audio/opus/tables/log-n400.csv`.
pub const LOG_N_400: [i32; CELT_NUM_BANDS] = [
    0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 21, 21, 24, 29, 34, 36,
];

/// Cumulative Table 55 band edges in *short-MDCT* bins
/// (`eBands`-style): `band_edge(i)` is the first bin of band `i`, and
/// `band_edge(21) = 100` is one past the last coded bin.
#[must_use]
pub fn band_edge(band: usize) -> i32 {
    let mut acc = 0i32;
    let mut b = 0usize;
    while b < band && b < CELT_NUM_BANDS {
        acc += celt_band_bins_per_channel(b, CeltFrameSize::Ms2_5).unwrap_or(0) as i32;
        b += 1;
    }
    acc
}

/// Width of `band` in short-MDCT bins (Table 55).
#[must_use]
pub fn band_width(band: usize) -> i32 {
    celt_band_bins_per_channel(band, CeltFrameSize::Ms2_5).unwrap_or(0) as i32
}

/// §4.3.4.1 pseudo-pulse → pulse-count expansion: indices 0–7 map to
/// themselves; each subsequent group of eight doubles the step.
#[must_use]
pub fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        (8 + (i & 7)) << ((i >> 3) - 1)
    }
}

/// The §4.3.4.1 cost-cache run for `(band, lm)`, where `lm` is the
/// *true* LM in `-1..=3` (the LM-major row is `lm + 1`). Returns the
/// run slice `[maxK, qbits[1..=maxK]]`, or an empty slice for a
/// sentinel tuple (never consulted by the allocator).
#[must_use]
pub fn cache_run(band: usize, lm: i32) -> &'static [u8] {
    let row = (lm + 1) as usize;
    let Some(&off) = CACHE_INDEX50.get(row * CELT_NUM_BANDS + band) else {
        return &[];
    };
    if off < 0 {
        return &[];
    }
    let off = off as usize;
    let max_k = CACHE_BITS50[off] as usize;
    &CACHE_BITS50[off..=off + max_k]
}

/// §4.3.4.1 *Bits to Pulses*: the pseudo-pulse count whose cached cost
/// best matches `bits` (1/8 bits) for `(band, lm)`; `lm ∈ -1..=3`.
#[must_use]
pub fn bits2pulses(band: usize, lm: i32, bits: i32) -> i32 {
    let cache = cache_run(band, lm);
    if cache.is_empty() {
        return 0;
    }
    let mut lo: i32 = 0;
    let mut hi: i32 = cache[0] as i32;
    let bits = bits - 1;
    // LOG_MAX_PSEUDO = 6 bisection steps over the monotone cost curve.
    for _ in 0..6 {
        let mid = (lo + hi + 1) >> 1;
        if cache[mid as usize] as i32 >= bits {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lo_cost = if lo == 0 {
        -1
    } else {
        cache[lo as usize] as i32
    };
    if bits - lo_cost <= cache[hi as usize] as i32 - bits {
        lo
    } else {
        hi
    }
}

/// §4.3.4.1 inverse: the cached cost (1/8 bits) of `pulses`
/// pseudo-pulses for `(band, lm)`; `lm ∈ -1..=3`.
#[must_use]
pub fn pulses2bits(band: usize, lm: i32, pulses: i32) -> i32 {
    let cache = cache_run(band, lm);
    if pulses == 0 || cache.is_empty() {
        0
    } else {
        cache[pulses as usize] as i32 + 1
    }
}

/// The result of the §4.3.3 implicit allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Allocation {
    /// Number of coded bands after skip decoding: bands
    /// `start..coded_bands` receive PVQ bits; bands
    /// `coded_bands..end` receive only fine energy.
    pub coded_bands: usize,
    /// Per-band PVQ budget in 1/8 bits (`bits[]`).
    pub pulses: [i32; CELT_NUM_BANDS],
    /// Per-band fine-energy bits per channel (`ebits[]`).
    pub fine_bits: [i32; CELT_NUM_BANDS],
    /// Per-band fine-energy priority for the §4.3.2.3 final-bit
    /// backfill (`fine_priority[]`; priority 0 bands are filled first).
    pub fine_priority: [bool; CELT_NUM_BANDS],
    /// First intensity-stereo-coded band (bands `>= intensity` are
    /// intensity-coded; `0` = intensity off).
    pub intensity: usize,
    /// Dual-stereo flag.
    pub dual_stereo: bool,
    /// Left-over 1/8 bits carried into the §4.3.4 band decode's
    /// running rebalancing.
    pub balance: i32,
}

/// Parameters shared by the §4.3.3 allocation stages.
struct AllocCtx {
    start: usize,
    end: usize,
    channels: i32,
    lm: i32,
}

/// §4.3.3 implicit allocation, decode side (normative
/// `compute_allocation` with `encode = 0`).
///
/// * `rd` — the frame's range decoder, positioned right after the trim
///   symbol; the skip / intensity / dual-stereo symbols are read from
///   it at the exact points the computation defines.
/// * `start..end` — the coded band range.
/// * `offsets` — per-band dynalloc boosts in 1/8 bits (§4.3.3 band
///   boost decode).
/// * `cap` — per-band caps in 1/8 bits (`init_caps`,
///   [`crate::celt_cache_caps50::cap_for_band_bits`]).
/// * `alloc_trim` — the decoded Table 58 trim (0–10; 5 = neutral).
/// * `total` — the available budget in 1/8 bits (frame bits minus
///   `ec_tell_frac()` minus 1, minus the anti-collapse reservation).
/// * `channels` — 1 or 2; `lm` — the frame-size shift (0–3).
#[allow(clippy::too_many_arguments)]
pub fn compute_allocation_decode(
    rd: &mut RangeDecoder<'_>,
    start: usize,
    end: usize,
    offsets: &[i32; CELT_NUM_BANDS],
    cap: &[i32; CELT_NUM_BANDS],
    alloc_trim: i32,
    total: i32,
    channels: usize,
    lm: i32,
) -> Allocation {
    let ctx = AllocCtx {
        start,
        end,
        channels: channels as i32,
        lm,
    };
    let c = ctx.channels;
    let mut total = total.max(0);
    let mut skip_start = start;

    // Reserve a bit to signal the end of manually skipped bands.
    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;

    // Reserve bits for the intensity and dual-stereo parameters.
    let mut intensity_rsv: i32 = 0;
    let mut dual_stereo_rsv: i32 = 0;
    if c == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[end - start] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
            total -= dual_stereo_rsv;
        }
    }

    let mut thresh = [0i32; CELT_NUM_BANDS];
    let mut trim_offset = [0i32; CELT_NUM_BANDS];
    for j in start..end {
        let width = band_width(j);
        // Below this threshold, we're sure not to allocate any PVQ
        // bits.
        thresh[j] = (c << BITRES).max((((3 * width) << lm) << BITRES) >> 4);
        // Tilt of the allocation curve.
        trim_offset[j] =
            (c * width * (alloc_trim - 5 - lm) * (end as i32 - j as i32 - 1) * (1 << (lm + 3)))
                >> 6;
        // Less resolution for single-coefficient bands.
        if width << lm == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }

    // Coarse search over the Table 57 quality rows.
    let mut lo: i32 = 1;
    let mut hi: i32 = N_ALLOC_VECTORS - 1;
    loop {
        let mut done = false;
        let mut psum = 0i32;
        let mid = (lo + hi) >> 1;
        for j in (start..end).rev() {
            let n = band_width(j);
            let mut bitsj = (c * n * STATIC_ALLOC[j][mid as usize] as i32) << lm >> 2;
            if bitsj > 0 {
                bitsj = 0.max(bitsj + trim_offset[j]);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += bitsj.min(cap[j]);
            } else if bitsj >= c << BITRES {
                psum += c << BITRES;
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
        if lo > hi {
            break;
        }
    }
    hi = lo;
    lo -= 1;

    // Per-band endpoints of the interpolation interval.
    let mut bits1 = [0i32; CELT_NUM_BANDS];
    let mut bits2 = [0i32; CELT_NUM_BANDS];
    for j in start..end {
        let n = band_width(j);
        let mut bits1j = (c * n * STATIC_ALLOC[j][lo as usize] as i32) << lm >> 2;
        let mut bits2j = if hi >= N_ALLOC_VECTORS {
            cap[j]
        } else {
            (c * n * STATIC_ALLOC[j][hi as usize] as i32) << lm >> 2
        };
        if bits1j > 0 {
            bits1j = 0.max(bits1j + trim_offset[j]);
        }
        if bits2j > 0 {
            bits2j = 0.max(bits2j + trim_offset[j]);
        }
        if lo > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = 0.max(bits2j - bits1j);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }

    interp_bits2pulses(
        rd,
        &ctx,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        skip_rsv,
        intensity_rsv,
        dual_stereo_rsv,
    )
}

/// The §4.3.3 interpolation + skip/intensity/dual decode + fine-energy
/// split (normative `interp_bits2pulses`, decode side).
#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses(
    rd: &mut RangeDecoder<'_>,
    ctx: &AllocCtx,
    skip_start: usize,
    bits1: &[i32; CELT_NUM_BANDS],
    bits2: &[i32; CELT_NUM_BANDS],
    thresh: &[i32; CELT_NUM_BANDS],
    cap: &[i32; CELT_NUM_BANDS],
    mut total: i32,
    skip_rsv: i32,
    mut intensity_rsv: i32,
    mut dual_stereo_rsv: i32,
) -> Allocation {
    let (start, end, c, lm) = (ctx.start, ctx.end, ctx.channels, ctx.lm);
    let stereo = i32::from(c > 1);
    let alloc_floor = c << BITRES;
    let log_m = lm << BITRES;

    // Fine-grained bisection of the interpolation fraction.
    let mut lo: i32 = 0;
    let mut hi: i32 = 1 << ALLOC_STEPS;
    for _ in 0..ALLOC_STEPS {
        let mid = (lo + hi) >> 1;
        let mut psum = 0i32;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> ALLOC_STEPS);
            if tmp >= thresh[j] || done {
                done = true;
                psum += tmp.min(cap[j]);
            } else if tmp >= alloc_floor {
                psum += alloc_floor;
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let mut psum = 0i32;
    let mut done = false;
    let mut bits = [0i32; CELT_NUM_BANDS];
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> ALLOC_STEPS);
        if tmp < thresh[j] && !done {
            if tmp >= alloc_floor {
                tmp = alloc_floor;
            } else {
                tmp = 0;
            }
        } else {
            done = true;
        }
        let tmp = tmp.min(cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    // Decide which bands to skip, working backwards from the end.
    let mut coded_bands = end;
    loop {
        let j = coded_bands - 1;
        // Never skip the first band, nor a band boosted by dynalloc.
        if j <= skip_start {
            // Give the bit we reserved to end skipping back.
            total += skip_rsv;
            break;
        }
        // Left-over bits this band would receive (including bits
        // stolen back from higher, skipped bands).
        let mut left = total - psum;
        let coded_width = band_edge(coded_bands) - band_edge(start);
        let percoeff = left / coded_width;
        left -= coded_width * percoeff;
        let rem = 0.max(left - (band_edge(j) - band_edge(start)));
        let band_width_bins = band_edge(coded_bands) - band_edge(j);
        let mut band_bits = bits[j] + percoeff * band_width_bins + rem;
        // Only code a skip decision above the threshold; otherwise the
        // band is force-skipped.
        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            if rd.dec_bit_logp(1) == 1 {
                break;
            }
            // We used a bit to skip this band.
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        // Reclaim the bits originally allocated to this band.
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = LOG2_FRAC_TABLE[j - start] as i32;
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            // Enough for a fine-energy bit per channel.
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }
    debug_assert!(coded_bands > start);

    // Decode the intensity and dual-stereo parameters.
    let mut intensity: usize = 0;
    if intensity_rsv > 0 {
        intensity = start
            + rd.dec_uint((coded_bands + 1 - start) as u32)
                .unwrap_or_default() as usize;
    }
    if intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    let dual_stereo = dual_stereo_rsv > 0 && rd.dec_bit_logp(1) == 1;

    // Allocate the remaining bits.
    let mut left = total - psum;
    let coded_width = band_edge(coded_bands) - band_edge(start);
    let percoeff = left / coded_width;
    left -= coded_width * percoeff;
    for (j, b) in bits.iter_mut().enumerate().take(coded_bands).skip(start) {
        *b += percoeff * band_width(j);
    }
    for (j, b) in bits.iter_mut().enumerate().take(coded_bands).skip(start) {
        let tmp = left.min(band_width(j));
        *b += tmp;
        left -= tmp;
    }

    // Split each band's budget into fine energy and PVQ, carrying the
    // over-cap excess forward as balance.
    let mut ebits = [0i32; CELT_NUM_BANDS];
    let mut fine_priority = [false; CELT_NUM_BANDS];
    let mut balance = 0i32;
    for j in start..coded_bands {
        let n0 = band_width(j);
        let n = n0 << lm;
        debug_assert!(bits[j] >= 0);
        bits[j] += balance;
        let mut excess;

        if n > 1 {
            excess = 0.max(bits[j] - cap[j]);
            bits[j] -= excess;

            // Compensate for the extra degree of freedom in stereo.
            let den = c * n + i32::from(c == 2 && n > 2 && !dual_stereo && j < intensity);
            let nclog_n = den * (LOG_N_400[j] + log_m);

            // Offset the fine bits by log2(N)/2 + FINE_OFFSET relative
            // to their fair share.
            let mut offset = (nclog_n >> 1) - den * FINE_OFFSET;

            // N=2 is the only point that doesn't match the curve.
            if n == 2 {
                offset += den << BITRES >> 2;
            }

            // Bias the second and third fine-energy bit thresholds.
            if bits[j] + offset < (den * 2) << BITRES {
                offset += nclog_n >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nclog_n >> 3;
            }

            // Divide with rounding.
            ebits[j] = 0.max((bits[j] + offset + (den << (BITRES - 1))) / (den << BITRES));

            // Make sure not to bust.
            if c * ebits[j] > bits[j] >> BITRES {
                ebits[j] = bits[j] >> stereo >> BITRES;
            }

            // More than MAX_FINE_BITS is useless for PVQ.
            ebits[j] = ebits[j].min(MAX_FINE_BITS);

            // Rounded-down / capped bands become candidates for the
            // final fine-energy pass.
            fine_priority[j] = ebits[j] * (den << BITRES) >= bits[j] + offset;

            // Remove the fine bits; the rest goes to PVQ.
            bits[j] -= (c * ebits[j]) << BITRES;
        } else {
            // N = 1: all bits go to fine energy except a single sign
            // bit per channel.
            excess = 0.max(bits[j] - (c << BITRES));
            bits[j] -= excess;
            ebits[j] = 0;
            fine_priority[j] = true;
        }

        // Fine energy can't use the §4.3.4 band-decode rebalancing;
        // re-balance the excess here instead.
        if excess > 0 {
            let extra_fine = (excess >> (stereo + BITRES as i32)).min(MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = (extra_fine * c) << BITRES;
            fine_priority[j] = extra_bits >= excess - balance;
            excess -= extra_bits;
        }
        balance = excess;

        debug_assert!(bits[j] >= 0);
        debug_assert!(ebits[j] >= 0);
    }

    // Skipped bands use all their bits for fine energy.
    for j in coded_bands..end {
        ebits[j] = bits[j] >> stereo >> BITRES;
        debug_assert_eq!((c * ebits[j]) << BITRES, bits[j]);
        bits[j] = 0;
        fine_priority[j] = ebits[j] < 1;
    }

    Allocation {
        coded_bands,
        pulses: bits,
        fine_bits: ebits,
        fine_priority,
        intensity,
        dual_stereo,
        balance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::celt_cache_caps50::{cap_for_band_bits, CacheCapsStereo};

    fn caps_for(lm: u32, channels: u32) -> [i32; CELT_NUM_BANDS] {
        let stereo = if channels == 2 {
            CacheCapsStereo::Stereo
        } else {
            CacheCapsStereo::Mono
        };
        let mut cap = [0i32; CELT_NUM_BANDS];
        for (band, c) in cap.iter_mut().enumerate() {
            let bins = (band_width(band) << lm) as u32;
            *c = cap_for_band_bits(lm, stereo, band as u32, channels, bins).unwrap() as i32;
        }
        cap
    }

    #[test]
    fn band_edges_match_table55() {
        assert_eq!(band_edge(0), 0);
        assert_eq!(band_edge(1), 1);
        assert_eq!(band_edge(17), 40);
        assert_eq!(band_edge(21), 100);
    }

    #[test]
    fn get_pulses_expands_pseudo_pulses() {
        // 0..8 map to themselves; the next octave steps by 2, etc.
        for i in 0..8 {
            assert_eq!(get_pulses(i), i);
        }
        assert_eq!(get_pulses(8), 8);
        assert_eq!(get_pulses(9), 9);
        assert_eq!(get_pulses(15), 15);
        assert_eq!(get_pulses(16), 16);
        assert_eq!(get_pulses(17), 18);
        assert_eq!(get_pulses(24), 32);
        assert_eq!(get_pulses(40), 128);
    }

    #[test]
    fn bits2pulses_and_pulses2bits_are_consistent() {
        // For every non-sentinel (band, lm) and every pseudo-pulse
        // count, converting the exact cached cost back must return the
        // same pseudo-pulse count (the search picks the nearest match,
        // and the exact cost is trivially nearest to itself).
        for lm in -1i32..=3 {
            for band in 0..CELT_NUM_BANDS {
                let run = cache_run(band, lm);
                if run.is_empty() {
                    continue;
                }
                let max_k = run[0] as i32;
                for q in 1..=max_k {
                    let cost = pulses2bits(band, lm, q);
                    // The cost curve need not be strictly increasing
                    // (the flat N = 1 run), so the search can land on
                    // a smaller pseudo-pulse count with the identical
                    // cost; the recovered *cost* must be exact.
                    let q_back = bits2pulses(band, lm, cost);
                    assert_eq!(
                        pulses2bits(band, lm, q_back),
                        cost,
                        "band {band} lm {lm} q {q} cost {cost} q_back {q_back}"
                    );
                    assert!(q_back <= q);
                }
            }
        }
    }

    #[test]
    fn pulses2bits_zero_is_free() {
        assert_eq!(pulses2bits(5, 2, 0), 0);
    }

    #[test]
    fn allocation_respects_caps_and_budget() {
        // A generously-sized frame: every band's PVQ + fine budget must
        // stay within its cap + balance accounting, and the coded-band
        // count within range.
        let buf = [0x5Au8; 200];
        let mut rd = RangeDecoder::new(&buf);
        let cap = caps_for(3, 1);
        let offsets = [0i32; CELT_NUM_BANDS];
        let total = 200 * 64 - 8;
        let alloc = compute_allocation_decode(&mut rd, 0, 21, &offsets, &cap, 5, total, 1, 3);
        assert!(alloc.coded_bands > 0 && alloc.coded_bands <= 21);
        let mut spent = alloc.balance;
        for j in 0..alloc.coded_bands {
            assert!(alloc.pulses[j] >= 0);
            assert!(alloc.fine_bits[j] >= 0);
            assert!(alloc.fine_bits[j] <= MAX_FINE_BITS);
            spent += alloc.pulses[j] + (alloc.fine_bits[j] << BITRES);
        }
        assert!(spent <= total + (1 << BITRES), "spent {spent} > {total}");
        // Mono never decodes intensity / dual stereo.
        assert_eq!(alloc.intensity, 0);
        assert!(!alloc.dual_stereo);
    }

    #[test]
    fn tiny_budget_allocates_only_fine_floor() {
        // With almost no bits, every band collapses to the alloc floor
        // or zero and no skip bits are decodable.
        let buf = [0xFFu8; 4];
        let mut rd = RangeDecoder::new(&buf);
        let cap = caps_for(0, 1);
        let offsets = [0i32; CELT_NUM_BANDS];
        let alloc = compute_allocation_decode(&mut rd, 0, 21, &offsets, &cap, 5, 8, 1, 0);
        for j in 0..21 {
            assert!(alloc.pulses[j] <= 8, "band {j} got {}", alloc.pulses[j]);
        }
    }

    #[test]
    fn stereo_allocation_decodes_intensity_in_range() {
        let buf = [0xA7u8; 120];
        let mut rd = RangeDecoder::new(&buf);
        let cap = caps_for(3, 2);
        let offsets = [0i32; CELT_NUM_BANDS];
        let total = 120 * 64 - 8;
        let alloc = compute_allocation_decode(&mut rd, 0, 21, &offsets, &cap, 5, total, 2, 3);
        assert!(alloc.intensity <= alloc.coded_bands);
    }

    #[test]
    fn allocation_is_deterministic() {
        let buf = [0x3Cu8; 100];
        let cap = caps_for(2, 1);
        let offsets = [0i32; CELT_NUM_BANDS];
        let mut rd1 = RangeDecoder::new(&buf);
        let mut rd2 = RangeDecoder::new(&buf);
        let a = compute_allocation_decode(&mut rd1, 0, 21, &offsets, &cap, 4, 100 * 64, 1, 2);
        let b = compute_allocation_decode(&mut rd2, 0, 21, &offsets, &cap, 4, 100 * 64, 1, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn hybrid_window_allocates_only_coded_range() {
        // Hybrid: start = 17. Bands below start must stay zero.
        let buf = [0x91u8; 60];
        let mut rd = RangeDecoder::new(&buf);
        let cap = caps_for(1, 1);
        let offsets = [0i32; CELT_NUM_BANDS];
        let alloc = compute_allocation_decode(&mut rd, 17, 21, &offsets, &cap, 5, 60 * 64, 1, 1);
        for j in 0..17 {
            assert_eq!(alloc.pulses[j], 0);
            assert_eq!(alloc.fine_bits[j], 0);
        }
        assert!(alloc.coded_bands > 17);
    }
}
