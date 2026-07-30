//! CELT §4.3.3 bit allocation, **encode side** — the same implicit
//! allocation as [`crate::celt_rate_alloc`] with the three interleaved
//! symbols *written* instead of read: the band-skip flags (the only
//! encoder-choice part, decided by the listing's hysteresis threshold),
//! the intensity-stereo index, and the dual-stereo flag.
//!
//! Every arithmetic step is byte-for-byte the decode module's; only
//! the coder calls differ, so an encode followed by
//! [`crate::celt_rate_alloc::compute_allocation_decode`] on the
//! produced bytes reproduces the identical [`Allocation`] (pinned by
//! the roundtrip test).
//!
//! ## Provenance
//!
//! RFC 6716 §4.3.3 / §5.3.3 + the normative Appendix A reference
//! listing (staged `docs/audio/opus/rfc6716-opus.txt`, hash-verified).
//! No external library source was consulted.

use crate::celt_band_layout::CELT_NUM_BANDS;
use crate::celt_log2_frac_table::LOG2_FRAC_TABLE;
use crate::celt_rate_alloc::{band_edge, band_width, Allocation, BITRES, MAX_FINE_BITS};
use crate::celt_static_alloc::STATIC_ALLOC;
use crate::range_encoder::RangeEncoder;

const FINE_OFFSET: i32 = 21;
const ALLOC_STEPS: u32 = 6;
const N_ALLOC_VECTORS: i32 = 11;

/// §4.3.3 implicit allocation, encode side (normative
/// `compute_allocation` with `encode = 1`).
///
/// * `intensity` / `dual_stereo` — the encoder's stereo decisions
///   (absolute band index / flag); written into the stream where the
///   reservations allow and clamped exactly as the listing clamps
///   them. The returned [`Allocation`] carries the coded values.
/// * `prev` — the previous frame's coded-band count
///   (`lastCodedBands`, 0 on the first frame), feeding the skip
///   hysteresis.
#[allow(clippy::too_many_arguments)]
pub fn compute_allocation_encode(
    enc: &mut RangeEncoder,
    start: usize,
    end: usize,
    offsets: &[i32; CELT_NUM_BANDS],
    cap: &[i32; CELT_NUM_BANDS],
    alloc_trim: i32,
    total: i32,
    channels: usize,
    lm: i32,
    intensity: usize,
    dual_stereo: bool,
    prev: usize,
) -> Allocation {
    let c = channels as i32;
    let mut total = total.max(0);
    let mut skip_start = start;

    let skip_rsv = if total >= 1 << BITRES { 1 << BITRES } else { 0 };
    total -= skip_rsv;

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
        thresh[j] = (c << BITRES).max((((3 * width) << lm) << BITRES) >> 4);
        trim_offset[j] =
            (c * width * (alloc_trim - 5 - lm) * (end as i32 - j as i32 - 1) * (1 << (lm + 3)))
                >> 6;
        if width << lm == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }

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

    interp_bits2pulses_encode(
        enc,
        start,
        end,
        c,
        lm,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        skip_rsv,
        intensity_rsv,
        dual_stereo_rsv,
        intensity,
        dual_stereo,
        prev,
    )
}

#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses_encode(
    enc: &mut RangeEncoder,
    start: usize,
    end: usize,
    c: i32,
    lm: i32,
    skip_start: usize,
    bits1: &[i32; CELT_NUM_BANDS],
    bits2: &[i32; CELT_NUM_BANDS],
    thresh: &[i32; CELT_NUM_BANDS],
    cap: &[i32; CELT_NUM_BANDS],
    mut total: i32,
    skip_rsv: i32,
    mut intensity_rsv: i32,
    mut dual_stereo_rsv: i32,
    mut intensity: usize,
    mut dual_stereo: bool,
    prev: usize,
) -> Allocation {
    let stereo = i32::from(c > 1);
    let alloc_floor = c << BITRES;
    let log_m = lm << BITRES;

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
        if j <= skip_start {
            total += skip_rsv;
            break;
        }
        let mut left = total - psum;
        let coded_width = band_edge(coded_bands) - band_edge(start);
        let percoeff = left / coded_width;
        left -= coded_width * percoeff;
        let rem = 0.max(left - (band_edge(j) - band_edge(start)));
        let band_width_bins = band_edge(coded_bands) - band_edge(j);
        let mut band_bits = bits[j] + percoeff * band_width_bins + rem;
        if band_bits >= thresh[j].max(alloc_floor + (1 << BITRES)) {
            // The only non-mandatory part of the allocation: the
            // encoder chooses whether to stop skipping, with the
            // listing's hysteresis threshold.
            let keep_thresh =
                ((if j < prev { 7 } else { 9 }) * band_width_bins) << lm << BITRES >> 4;
            if band_bits > keep_thresh {
                enc.enc_bit_logp(true, 1);
                break;
            }
            enc.enc_bit_logp(false, 1);
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        psum -= bits[j] + intensity_rsv;
        if intensity_rsv > 0 {
            intensity_rsv = LOG2_FRAC_TABLE[j - start] as i32;
        }
        psum += intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }
    debug_assert!(coded_bands > start);

    // Code the intensity and dual-stereo parameters. (The caller
    // clamps intensity into [start, end]; re-clamp defensively so a
    // degenerate input cannot underflow the relative index.)
    if intensity_rsv > 0 {
        intensity = intensity.clamp(start, coded_bands);
        enc.enc_uint((intensity - start) as u32, (coded_bands + 1 - start) as u32);
    } else {
        intensity = 0;
    }
    if intensity <= start {
        total += dual_stereo_rsv;
        dual_stereo_rsv = 0;
    }
    if dual_stereo_rsv > 0 {
        enc.enc_bit_logp(dual_stereo, 1);
    } else {
        dual_stereo = false;
    }

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

    // Split each band's budget into fine energy and PVQ.
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

            let den = c * n + i32::from(c == 2 && n > 2 && !dual_stereo && j < intensity);
            let nclog_n = den * (crate::celt_rate_alloc::LOG_N_400[j] + log_m);
            let mut offset = (nclog_n >> 1) - den * FINE_OFFSET;
            if n == 2 {
                offset += den << BITRES >> 2;
            }
            if bits[j] + offset < (den * 2) << BITRES {
                offset += nclog_n >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nclog_n >> 3;
            }
            ebits[j] = 0.max((bits[j] + offset + (den << (BITRES - 1))) / (den << BITRES));
            if c * ebits[j] > bits[j] >> BITRES {
                ebits[j] = bits[j] >> stereo >> BITRES;
            }
            ebits[j] = ebits[j].min(MAX_FINE_BITS);
            fine_priority[j] = ebits[j] * (den << BITRES) >= bits[j] + offset;
            bits[j] -= (c * ebits[j]) << BITRES;
        } else {
            excess = 0.max(bits[j] - (c << BITRES));
            bits[j] -= excess;
            ebits[j] = 0;
            fine_priority[j] = true;
        }

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
    use crate::celt_rate_alloc::compute_allocation_decode;
    use crate::range_decoder::RangeDecoder;

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
    fn allocation_encode_decode_roundtrip_across_the_parameter_surface() {
        // For a grid of (channels, LM, budget, trim, start) the encoded
        // skip/intensity/dual symbols must decode into the identical
        // Allocation.
        for &channels in &[1usize, 2] {
            for lm in 0..4i32 {
                for &frame_bytes in &[20i32, 60, 160, 400] {
                    for &(start, end) in &[(0usize, 21usize), (0, 17), (17, 21)] {
                        let cap = caps_for(lm as u32, channels as u32);
                        let mut offsets = [0i32; CELT_NUM_BANDS];
                        offsets[5] = if frame_bytes > 100 { 64 } else { 0 };
                        let total = frame_bytes * 64 - 20;
                        let trim = 5;
                        let intensity = if channels == 2 {
                            12.clamp(start, end)
                        } else {
                            0
                        };
                        let mut enc = RangeEncoder::new();
                        let a_enc = compute_allocation_encode(
                            &mut enc,
                            start,
                            end,
                            &offsets,
                            &cap,
                            trim,
                            total,
                            channels,
                            lm,
                            intensity,
                            channels == 2,
                            0,
                        );
                        let buf = enc.finish();
                        let mut rd = RangeDecoder::new(&buf);
                        let a_dec = compute_allocation_decode(
                            &mut rd, start, end, &offsets, &cap, trim, total, channels, lm,
                        );
                        assert_eq!(
                            a_enc, a_dec,
                            "channels={channels} lm={lm} bytes={frame_bytes} start={start}"
                        );
                    }
                }
            }
        }
    }
}
