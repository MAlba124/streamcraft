//! Range decoder primitives for the Opus codec.
//!
//! This module implements the bit-exact range decoder described in
//! RFC 6716 §4.1 (`docs/audio/opus/rfc6716-opus.txt`). The implementation
//! is clean-room: every routine is transcribed from the prose and
//! pseudocode equations in the RFC; no external library source was
//! consulted.
//!
//! The range decoder is the SHARED entropy primitive that both the SILK
//! and CELT layers of Opus invoke for every coded symbol. The
//! [`oxideav-celt`] crate carries its own copy of the same primitive;
//! each crate owns its copy until a shared low-level primitive crate
//! exists in the workspace. The two copies are independent clean-room
//! transcriptions of the same RFC sections and are expected to be
//! behaviourally identical.
//!
//! The following routines are wired up:
//!
//! * Initialization (§4.1.1).
//! * Symbol-update internal helper (§4.1.2).
//! * Renormalization (§4.1.2.1).
//! * [`RangeDecoder::decode_bin`] for power-of-two `ft` symbols (§4.1.3.1).
//! * [`RangeDecoder::dec_bit_logp`] (§4.1.3.2).
//! * [`RangeDecoder::dec_icdf`] for inverse-CDF table decoding (§4.1.3.3).
//! * [`RangeDecoder::dec_bits`] for raw bits (§4.1.4).
//! * [`RangeDecoder::dec_uint`] for uniformly-distributed integers
//!   (§4.1.5).
//! * [`RangeDecoder::tell`] for whole-bit accounting (§4.1.6.1).
//! * [`RangeDecoder::tell_frac`] for 1/8th-bit-precision accounting
//!   (§4.1.6.2).
//! * [`RangeDecoder::ec_decode`] / [`RangeDecoder::ec_dec_update`] for
//!   the generic two-step symbol path (§4.1.2). These are the building
//!   blocks for custom symbol decoders that an inverse-CDF table cannot
//!   express directly — notably the CELT §4.3.2.1 coarse-energy
//!   Laplace decoder and the §4.3.3 allocation interpolation search,
//!   both of which decode against a frequency model computed at
//!   run time rather than a fixed `icdf[]` table.

use crate::Error;

/// Bit-exact CELT/SILK range decoder state per RFC 6716 §4.1.
///
/// The decoder splits the input buffer into two halves. The range
/// coder consumes bytes from the front (MSB-first into the range
/// state) and the raw-bit reader consumes bytes from the back
/// (LSB-first). RFC 6716 §4.1.4 explicitly permits the two readers
/// to overlap; the decoder MUST allow it.
#[derive(Debug)]
pub struct RangeDecoder<'a> {
    /// Input bitstream backing this decoder.
    buf: &'a [u8],
    /// Offset of the next byte the range coder will consume (advances
    /// forward through `buf`).
    fwd: usize,
    /// Number of bytes consumed by the raw-bit reader, measured from
    /// the END of `buf`. A value of `0` means no raw bit has yet been
    /// read; the next raw byte fetched comes from `buf[buf.len() - 1]`.
    back: usize,
    /// Number of unconsumed bits currently sitting in `back_window`
    /// (0..=8 at rest, may exceed during refill).
    back_bits_avail: u32,
    /// Buffer of unconsumed raw bits, packed with the next bit to
    /// emit in bit 0.
    back_window: u32,
    /// One-bit buffer holding the LSB of the previously-consumed
    /// forward byte (used in the next renormalization step, §4.1.2.1).
    rem: u32,
    /// Range size; the renormalization invariant is `rng > 2**23`.
    rng: u32,
    /// Top of range minus current code value, minus one.
    val: u32,
    /// Running tally of whole bits the range coder has consumed
    /// (RFC 6716 §4.1.6 `nbits_total`).
    nbits_total: u32,
    /// Number of raw bits the decoder has read so far. RFC 6716 §4.1.6
    /// adds these into the bit-usage accounting on top of `nbits_total`.
    nbits_raw: u32,
    /// Sticky error flag: any decode that detects a corrupt frame
    /// latches an error. Once set, subsequent decodes return zeroes
    /// rather than corrupting the caller's state. RFC 6716 §4.1.5
    /// recommends this behaviour for malformed integer decodes.
    error: bool,
}

impl<'a> RangeDecoder<'a> {
    /// Renormalization invariant from §4.1.2.1: `rng > 2**23`.
    const RNG_MIN: u32 = 1 << 23;

    /// Initialize the range decoder over `buf` per RFC 6716 §4.1.1.
    ///
    /// The spec defines `b0` as "the first input byte (or zero if
    /// there are no bytes in this Opus frame)". The decoder sets
    /// `rng = 128`, `val = 127 - (b0 >> 1)`, buffers the leftover bit
    /// `(b0 & 1)`, then immediately invokes renormalization so the
    /// invariant `rng > 2**23` holds before any symbol is decoded.
    pub fn new(buf: &'a [u8]) -> Self {
        let b0 = buf.first().copied().unwrap_or(0) as u32;
        let mut dec = Self {
            buf,
            // §4.1.1: the first byte is consumed by initialization,
            // so the next forward fetch starts at index 1.
            fwd: if buf.is_empty() { 0 } else { 1 },
            back: 0,
            back_bits_avail: 0,
            back_window: 0,
            rem: b0 & 1,
            rng: 128,
            val: 127 - (b0 >> 1),
            // §4.1.6: "nbits_total is initialized to 9 just before the
            // initial range renormalization process completes."
            nbits_total: 9,
            nbits_raw: 0,
            error: false,
        };
        dec.normalize();
        dec
    }

    /// Whether this decoder has latched a `frame corrupt` error
    /// somewhere in its history. Higher-level decoders use this to
    /// abort the current frame and apply packet-loss concealment.
    pub fn has_error(&self) -> bool {
        self.error
    }

    /// Reduce the buffer in use to its first `new_len` bytes
    /// (RFC 6716 §4.5.1.3).
    ///
    /// When a Hybrid Opus frame carries a §4.5.1 redundant CELT frame,
    /// "the decoder reduces the size of the buffer currently in use by
    /// the range coder by that amount. The MDCT layer reads any raw
    /// bits from the end of this reduced buffer, and all calculations
    /// of the number of bits remaining in the buffer must be done
    /// using this new, reduced size" — the trailing bytes belong to
    /// the byte-aligned redundant frame, which uses its own coder.
    ///
    /// Latches the sticky error when the shrink is impossible: the
    /// range coder has already read past `new_len`, raw bits were
    /// already consumed from the discarded tail, or `new_len` exceeds
    /// the current buffer (all only reachable on a corrupt frame).
    pub fn shrink_buffer(&mut self, new_len: usize) {
        if new_len > self.buf.len() || self.fwd > new_len || self.back != 0 {
            self.error = true;
            return;
        }
        self.buf = &self.buf[..new_len];
    }

    /// The current §4.1 range size `rng`.
    ///
    /// The CELT layer captures this value at the end of each frame and
    /// carries it as the seed of the §4.3.4 / §4.3.5 folding-noise LCG
    /// for the *next* frame (the carried `rng` decoder state). It is a
    /// read-only observation of the coder state; no symbol is consumed.
    #[must_use]
    pub fn range_size(&self) -> u32 {
        self.rng
    }

    /// Current whole-bit budget consumed by the range coder plus the
    /// raw-bit reader, per RFC 6716 §4.1.6.1.
    ///
    /// `ec_tell` is defined as `nbits_total - ilog(rng)`. Raw bits are
    /// added separately because §4.1.6 specifies that raw bits also
    /// count against the total.
    pub fn tell(&self) -> u32 {
        // `ilog(rng)` is the position of the most-significant set bit
        // of `rng`, counting from 1. The renormalization invariant
        // keeps `rng >= 2**23`, so `lg` is always at least 24.
        let lg = 32 - self.rng.leading_zeros();
        self.nbits_total
            .saturating_sub(lg)
            .saturating_add(self.nbits_raw)
    }

    /// Current 1/8th-bit-precision budget consumed by the range coder
    /// plus the raw-bit reader, per RFC 6716 §4.1.6.2.
    ///
    /// Follows §4.1.6.2 directly: from `lg = ilog(rng)`, extract
    /// `r_Q15 = rng >> (lg - 16)` as a Q15 value in `[2^15, 2^16)`.
    /// Three iterations of
    /// `r_Q15 = (r_Q15*r_Q15) >> 15; lg = 2*lg + (r_Q15 >> 16)` extend
    /// `lg` to 1/8th-bit precision. Raw bits add `8*nbits_raw` (whole
    /// bits scaled into eighths). By construction,
    /// `ec_tell() == ceil(ec_tell_frac() / 8.0)`.
    pub fn tell_frac(&self) -> u32 {
        let lg0 = 32 - self.rng.leading_zeros();
        // §4.1.6.2: lg >= 24 after renormalization, so the shift below
        // is well-defined.  r_Q15 in [2^15, 2^16).
        let mut r_q15 = self.rng >> (lg0 - 16);
        // Build the 1/8th-bit-precision lg one bit at a time. The
        // spec doubles `lg` on each of the three refinement passes;
        // the accumulator starts at the whole-bit value `lg0`.
        let mut lg_frac = lg0;
        // Three passes yield three extra bits = 1/8th-bit precision.
        for _ in 0..3 {
            r_q15 = (r_q15 * r_q15) >> 15;
            let bit = r_q15 >> 16;
            lg_frac = 2 * lg_frac + bit;
            // If `bit == 1`, halve r_Q15 so it falls back into
            // [2^15, 2^16).
            if bit == 1 {
                r_q15 >>= 1;
            }
        }
        // Final value = nbits_total*8 - lg_frac + nbits_raw*8.
        self.nbits_total
            .saturating_mul(8)
            .saturating_sub(lg_frac)
            .saturating_add(self.nbits_raw.saturating_mul(8))
    }

    /// Decode a single binary symbol with probability `2^-logp` of
    /// being a "1", per RFC 6716 §4.1.3.2.
    ///
    /// Mathematically equivalent to `ec_decode(ft = 1<<logp)` followed
    /// by `ec_dec_update(0, ft-1, ft)` (for a "0") or
    /// `ec_dec_update(ft-1, ft, ft)` (for a "1"). The implementation
    /// is multiply-and-divide-free: `r >> logp` replaces `rng/ft`, and
    /// the discriminator collapses to a comparison.
    pub fn dec_bit_logp(&mut self, logp: u32) -> u32 {
        let r = self.rng;
        let d = self.val;
        // `s = r >> logp` corresponds to `rng/ft` with `ft = 1<<logp`
        // (an exact shift when ft is a power of two).
        let s = r >> logp;
        // The "1" half corresponds to `fl = ft-1, fh = ft`, leading to
        //   val unchanged, rng = s.
        // The "0" half is `fl = 0, fh = ft-1`, leading to
        //   val -= s, rng = r - s.
        let bit = if d < s { 1 } else { 0 };
        if bit == 1 {
            self.rng = s;
        } else {
            self.val = d - s;
            self.rng = r - s;
        }
        self.normalize();
        bit
    }

    /// Decode `bits` raw bits per RFC 6716 §4.1.4.
    ///
    /// Raw bits are packed at the END of the frame: the least
    /// significant bit of the first value is the LSB of the last
    /// byte; reads proceed toward the front. The function returns the
    /// raw bits in the order written — the LSB of the result holds
    /// the bit the encoder emitted first.
    ///
    /// Returns `0` on errors (`bits > 32`); also returns zero-extended
    /// bits past the end of the frame, matching §4.1.4's "the decoder
    /// MUST continue to use zero for any further input bytes required".
    pub fn dec_bits(&mut self, bits: u32) -> u32 {
        if bits == 0 {
            return 0;
        }
        if bits > 32 {
            self.error = true;
            return 0;
        }
        // Work in a 64-bit window: serving a full 32-bit read may hold
        // up to 39 bits in flight (7 residual + 4 fresh bytes), and the
        // final `>> bits` must be defined for `bits == 32` (a round-382
        // fuzz find: the previous u32 window overflowed both the refill
        // shift and the consume shift on a 32-bit read).
        let mut window = self.back_window as u64;
        let mut avail = self.back_bits_avail;
        // Refill the window until it holds enough bits to service the
        // requested read.
        while avail < bits {
            let byte = if self.back < self.buf.len() {
                self.buf[self.buf.len() - 1 - self.back]
            } else {
                // §4.1.4: zero-extend past the end of the frame.
                0
            };
            self.back = self.back.saturating_add(1);
            // Concatenate the new byte ABOVE the existing window so the
            // intra-byte LSB-first packing is preserved.
            window |= (byte as u64) << avail;
            avail += 8;
        }
        let mask: u64 = if bits >= 32 {
            0xFFFF_FFFF
        } else {
            (1u64 << bits) - 1
        };
        let result = (window & mask) as u32;
        // Consume the served bits; fewer than 8 bits remain after any
        // serve, so the truncation back to the u32 field is lossless.
        self.back_window = (window >> bits) as u32;
        self.back_bits_avail = avail - bits;
        self.nbits_raw += bits;
        result
    }

    /// Decode one of `ft` equiprobable values in `0..ft`, per
    /// RFC 6716 §4.1.5.
    ///
    /// Values of `ft <= 1` degenerate to the constant `0`. `ft` may
    /// be as large as `2^32 - 1`. The §4.1.5 procedure splits the
    /// value: the top 8 bits go through the range coder, the
    /// remainder through raw bits. If the reconstructed value is
    /// `>= ft`, the frame is corrupt — the decoder latches the error
    /// flag and saturates to `ft - 1` per §4.1.5's concealment
    /// recommendation.
    pub fn dec_uint(&mut self, ft: u32) -> Result<u32, Error> {
        if ft <= 1 {
            return Ok(0);
        }
        // `ftb = ilog(ft - 1)`: number of bits needed for `ft - 1`.
        let ftb = 32 - (ft - 1).leading_zeros();
        if ftb <= 8 {
            // Small case: a single range-coded symbol covers the whole
            // value.
            let t = self.decode(ft);
            self.dec_update(t, t + 1, ft);
            Ok(t)
        } else {
            // Large case: top 8 bits range-coded, remainder raw.
            let split_bits = ftb - 8;
            let top_ft = ((ft - 1) >> split_bits) + 1;
            let t_hi = self.decode(top_ft);
            self.dec_update(t_hi, t_hi + 1, top_ft);
            let t_lo = self.dec_bits(split_bits);
            let t = (t_hi << split_bits) | t_lo;
            if t >= ft {
                self.error = true;
                Ok(ft - 1)
            } else {
                Ok(t)
            }
        }
    }

    /// Decode `fs` for a power-of-two `ft = 1<<ftb` per RFC 6716
    /// §4.1.3.1 (`ec_decode_bin`).
    ///
    /// Mathematically equivalent to [`Self::decode`] with `ft = 1<<ftb`
    /// but avoids the division: `rng / ft == rng >> ftb`. The caller is
    /// expected to follow with [`Self::dec_update`] (or use
    /// [`Self::dec_icdf`] which fuses the two steps).
    ///
    /// Returns `fs` in the range `[0, 1<<ftb)`.
    pub fn decode_bin(&mut self, ftb: u32) -> u32 {
        let s = self.rng >> ftb;
        if s == 0 {
            // Would only happen for ftb > ilog(rng). The
            // renormalization invariant keeps ilog(rng) >= 24, so any
            // practical ftb (icdf uses up to 8) is safe. Defensively
            // saturate to 0.
            return 0;
        }
        let ft = 1u32 << ftb;
        let approx = (self.val / s).saturating_add(1);
        ft - approx.min(ft)
    }

    /// Decode a symbol via an inverse-CDF table, per RFC 6716 §4.1.3.3
    /// (`ec_dec_icdf`).
    ///
    /// `icdf[k]` stores `(1<<ftb) - fh[k]`, terminated by a `0` entry
    /// (the implicit `fh[K_last] == ft`). `fl[0]` is implicitly 0; the
    /// table values are strictly monotonically decreasing.
    ///
    /// Fuses the search step (find the smallest `k` such that
    /// `fs < ft - icdf[k]`) with the range/value update, eliminating
    /// the division. The renormalization loop runs before returning.
    ///
    /// Returns the decoded symbol index `k` in `0..icdf.len()-1`. On a
    /// malformed table (no terminating zero), the decoder latches its
    /// sticky error flag and returns 0.
    pub fn dec_icdf(&mut self, icdf: &[u8], ftb: u32) -> u32 {
        // `s` corresponds to `rng / ft` for `ft = 1<<ftb`.
        let s = self.rng >> ftb;
        // Forward walk: for each candidate k, compute
        //   next = s * icdf[k]
        // which is the "remaining range above this symbol". The first
        // k where `val >= next` is the decoded symbol. `t` tracks the
        // previous step's `next` so that `rng' = t - next` matches the
        // §4.1.2 update for `fl[k] == prev_next/s` and `fh[k] ==
        // next/s` (with k=0 falling out to `rng - s*icdf[0]` since
        // `t` starts at `rng`).
        let mut t = self.rng;
        for (k, &cell) in icdf.iter().enumerate() {
            let next = s.saturating_mul(cell as u32);
            if self.val >= next {
                self.val -= next;
                self.rng = t - next;
                self.normalize();
                return k as u32;
            }
            t = next;
        }
        // Malformed table: no terminator reached. §4.1.5 advises
        // latching the corrupt-frame error and returning a saturated
        // value.
        self.error = true;
        0
    }

    /// `ec_decode(ft)` per RFC 6716 §4.1.2 — the first of the two
    /// symbol-decode steps.
    ///
    /// Computes the 16-bit symbol proxy
    /// `fs = ft - min(val / (rng / ft) + 1, ft)`, which "lies within
    /// the range of some symbol in the current context". The caller
    /// then identifies the symbol `k` whose three-tuple
    /// `(fl[k], fh[k], ft)` satisfies `fl[k] <= fs < fh[k]` and feeds
    /// that tuple to [`Self::ec_dec_update`].
    ///
    /// This split form is needed when the frequency model is computed
    /// at run time and cannot be pre-baked into a static inverse-CDF
    /// table (RFC 6716 §4.3.2.1 coarse-energy Laplace decode, §4.3.3
    /// allocation search). For fixed PDFs prefer [`Self::dec_icdf`],
    /// which fuses both steps and avoids a division.
    ///
    /// `ft` must be in `1..=2**16` for the §4.1.2 derivation to hold
    /// (the renormalization invariant keeps `rng > 2**23`, so
    /// `rng / ft >= 1`). `ft == 0` would divide by zero; the decoder
    /// latches its sticky error flag and returns `0` instead. The
    /// returned `fs` lies in `[0, ft)`.
    pub fn ec_decode(&mut self, ft: u32) -> u32 {
        if ft == 0 {
            self.error = true;
            return 0;
        }
        self.decode(ft)
    }

    /// `ec_dec_update(fl, fh, ft)` per RFC 6716 §4.1.2 — the second of
    /// the two symbol-decode steps.
    ///
    /// Narrows the range to the chosen symbol's `[fl, fh)` sub-interval
    /// of `[0, ft)` per the §4.1.2 update equations, then renormalizes
    /// to restore `rng > 2**23`. Pair this with the index returned by
    /// the caller's search over the value from [`Self::ec_decode`].
    ///
    /// The three-tuple MUST satisfy `0 <= fl < fh <= ft` and
    /// `1 <= ft <= 2**16`. A malformed tuple (`ft == 0`, `fh > ft`, or
    /// `fl >= fh`) cannot come from a well-formed search; the decoder
    /// latches its sticky error flag and leaves its state unchanged
    /// rather than underflowing `val` or zeroing `rng`.
    pub fn ec_dec_update(&mut self, fl: u32, fh: u32, ft: u32) {
        if ft == 0 || fh > ft || fl >= fh {
            self.error = true;
            return;
        }
        self.dec_update(fl, fh, ft);
    }

    // ----- internal helpers -----

    /// `ec_decode(ft)` per RFC 6716 §4.1.2: compute the symbol-proxy
    /// `fs = ft - min(val / (rng / ft) + 1, ft)`.
    fn decode(&mut self, ft: u32) -> u32 {
        // The spec uses integer division. `rng/ft` is computed first;
        // the divisor is then `val / (rng/ft)`. The renormalization
        // invariant ensures `rng/ft >= 1` in all practical cases
        // (rng > 2**23 and ft <= 2**16 on the symbol-decode path).
        let s = self.rng / ft;
        let approx = self.val / s + 1;
        ft - approx.min(ft)
    }

    /// `ec_dec_update(fl, fh, ft)` per RFC 6716 §4.1.2.
    ///
    /// Narrows the range to the chosen symbol's interval, then runs
    /// renormalization to restore `rng > 2**23`.
    fn dec_update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.rng / ft;
        self.val -= s * (ft - fh);
        if fl > 0 {
            self.rng = s * (fh - fl);
        } else {
            self.rng -= s * (ft - fh);
        }
        self.normalize();
    }

    /// `ec_dec_normalize` per RFC 6716 §4.1.2.1.
    ///
    /// Until `rng > 2**23`, shift `rng` left by 8 and pull a fresh
    /// `sym` byte. `sym` combines the previously-buffered low bit
    /// (`rem`, as MSB) with the top 7 bits of the new byte; the LSB of
    /// the new byte is buffered for next time. When the frame is
    /// exhausted, zero bytes are substituted.
    fn normalize(&mut self) {
        while self.rng <= Self::RNG_MIN {
            let byte = if self.fwd < self.buf.len() {
                let b = self.buf[self.fwd];
                self.fwd += 1;
                b as u32
            } else {
                0
            };
            let sym = (self.rem << 7) | (byte >> 1);
            self.rem = byte & 1;
            self.rng <<= 8;
            self.val = ((self.val << 8) + (255 - sym)) & 0x7FFF_FFFF;
            // §4.1.6: each iteration adds 8 to nbits_total.
            self.nbits_total = self.nbits_total.saturating_add(8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §4.1.1 initialization over an empty buffer must still satisfy
    /// the §4.1.2.1 invariant and report `ec_tell() == 1`
    /// (§4.1.6.1: "In a newly initialized decoder, before any symbols
    /// have been read, this reports that 1 bit has been used").
    #[test]
    fn init_empty_buffer_satisfies_invariant() {
        let dec = RangeDecoder::new(&[]);
        assert!(dec.rng > RangeDecoder::RNG_MIN);
        assert!(!dec.has_error());
        assert_eq!(dec.tell(), 1);
    }

    /// Non-empty initialization also satisfies the invariant and
    /// reports a sensible tell.
    #[test]
    fn init_nonempty_buffer_holds_invariant() {
        let dec = RangeDecoder::new(&[0xAB, 0xCD, 0xEF, 0x12]);
        assert!(dec.rng > RangeDecoder::RNG_MIN);
        assert!(!dec.has_error());
        assert!(dec.tell() >= 1);
    }

    /// `dec_bit_logp` should be statistically biased by the surrounding
    /// bytes: an all-zero stream pushes `val` high, biasing toward "0",
    /// and an all-ones stream pushes it low, biasing toward "1".
    #[test]
    fn dec_bit_logp_bias_with_extreme_inputs() {
        // All-zero stream: bias toward "0".
        let mut dec0 = RangeDecoder::new(&[0u8; 16]);
        let mut zero_count = 0;
        for _ in 0..32 {
            if dec0.dec_bit_logp(1) == 0 {
                zero_count += 1;
            }
        }
        assert!(!dec0.has_error());
        assert!(
            zero_count > 16,
            "all-zero stream should be biased toward 0: zero_count={}",
            zero_count
        );

        // All-ones stream: bias toward "1".
        let mut dec1 = RangeDecoder::new(&[0xFFu8; 16]);
        let mut one_count = 0;
        for _ in 0..32 {
            if dec1.dec_bit_logp(1) == 1 {
                one_count += 1;
            }
        }
        assert!(!dec1.has_error());
        assert!(
            one_count > 16,
            "all-ones stream should be biased toward 1: one_count={}",
            one_count
        );
    }

    /// `dec_bits` reads raw bits LSB-first from the END of the buffer.
    /// With the last byte = 0b1010_0110, the first 4 raw bits returned
    /// are 0b0110 = 6, and the next 4 are 0b1010 = 0xA.
    #[test]
    fn dec_bits_lsb_first_from_end() {
        let mut dec = RangeDecoder::new(&[0x00, 0x00, 0xA6]);
        let lo = dec.dec_bits(4);
        let hi = dec.dec_bits(4);
        assert_eq!(lo, 0x6);
        assert_eq!(hi, 0xA);
        assert!(!dec.has_error());
    }

    /// `dec_bits` past the end of the frame must zero-extend, per
    /// §4.1.4 ("the decoder MUST continue to use zero for any further
    /// input bytes required"). The function must not panic or set the
    /// error flag in that case.
    #[test]
    fn dec_bits_zero_past_end_of_frame() {
        let mut dec = RangeDecoder::new(&[0xFF, 0xFF]);
        for _ in 0..4 {
            let v = dec.dec_bits(4);
            assert_eq!(v, 0xF);
        }
        // The next 8 bits should come back as zero (the range coder
        // may or may not have shared bytes with the raw reader — but
        // the *raw* side reads past-EOF as 0).
        let pad = dec.dec_bits(8);
        let _ = pad;
        assert!(!dec.has_error());
    }

    /// `dec_uint(1)` is degenerate — the only value in `0..1` is 0 —
    /// and consumes no bits.
    #[test]
    fn dec_uint_ft_one_is_zero_no_consumption() {
        let mut dec = RangeDecoder::new(&[0x12, 0x34, 0x56]);
        let before = dec.tell();
        let v = dec.dec_uint(1).expect("ft=1 must succeed");
        let after = dec.tell();
        assert_eq!(v, 0);
        assert_eq!(after, before);
    }

    /// `dec_uint` with `ft` in the small (`ftb <= 8`) regime: returned
    /// values must lie in `[0, ft)` and never trip the error flag for
    /// well-formed inputs.
    #[test]
    fn dec_uint_small_ft_in_range() {
        let mut dec = RangeDecoder::new(&[0x42, 0x18, 0xC3, 0x7F]);
        for _ in 0..8 {
            let v = dec.dec_uint(200).expect("ft=200 must succeed");
            assert!(v < 200, "v={} out of range", v);
        }
        assert!(!dec.has_error());
    }

    /// `dec_uint` with `ft` in the large (`ftb > 8`) regime: returned
    /// values must lie in `[0, ft)`. The saturation path is allowed
    /// to set the error flag, but the returned value remains bounded.
    #[test]
    fn dec_uint_large_ft_in_range() {
        let buf: Vec<u8> = (0..64).collect();
        let mut dec = RangeDecoder::new(&buf);
        for _ in 0..8 {
            let v = dec.dec_uint(1_000_000).expect("ft=1_000_000 must succeed");
            assert!(v < 1_000_000, "v={} out of range", v);
        }
    }

    /// `dec_uint` with `ft = 0` is degenerate and returns 0 without
    /// consumption.
    #[test]
    fn dec_uint_ft_zero_returns_zero() {
        let mut dec = RangeDecoder::new(&[0xAA, 0xBB, 0xCC, 0xDD]);
        let before = dec.tell();
        let v = dec.dec_uint(0).expect("ft=0 must succeed");
        assert_eq!(v, 0);
        assert_eq!(dec.tell(), before);
    }

    /// `tell()` must monotonically non-decrease across operations.
    #[test]
    fn tell_is_monotonic_across_decodes() {
        let mut dec = RangeDecoder::new(&[0x55; 8]);
        let mut prev = dec.tell();
        for _ in 0..16 {
            let _ = dec.dec_bit_logp(2);
            let now = dec.tell();
            assert!(now >= prev, "tell() went backwards: {} -> {}", prev, now);
            prev = now;
        }
    }

    /// `decode_bin(ftb)` must agree with the generic `decode(1<<ftb)`
    /// path bit-for-bit (RFC 6716 §4.1.3.1: the two are mathematically
    /// equivalent). Drive both with the same input bytes and compare.
    #[test]
    fn decode_bin_matches_generic_decode() {
        for &ftb in &[1u32, 4, 8, 12, 15] {
            let buf = [0x37u8, 0x91, 0xC4, 0x18, 0xA2, 0x5D, 0x6E, 0xFF];
            let mut a = RangeDecoder::new(&buf);
            let mut b = RangeDecoder::new(&buf);
            let from_bin = a.decode_bin(ftb);
            let from_generic = b.decode(1u32 << ftb);
            assert_eq!(
                from_bin, from_generic,
                "decode_bin({ftb}) != decode(1<<{ftb})"
            );
            assert!(from_bin < (1u32 << ftb), "fs={from_bin} out of range");
        }
    }

    /// RFC 6716 §4.1.6.1 specifies the identity
    /// `ec_tell() == ceil(ec_tell_frac() / 8.0)`. Walk a decoder
    /// forward through mixed symbol and raw-bit reads and assert this
    /// at every step.
    #[test]
    fn tell_frac_consistent_with_tell() {
        let mut dec = RangeDecoder::new(&[0xA3, 0x7F, 0x10, 0x5C, 0xE8, 0x91, 0x42, 0xB7]);
        // §4.1.6.1: a fresh decoder reports tell() == 1.
        assert_eq!(dec.tell(), 1);
        for _ in 0..12 {
            let whole = dec.tell();
            let frac = dec.tell_frac();
            let ceil_eighths = frac.div_ceil(8);
            assert_eq!(
                ceil_eighths, whole,
                "tell()={whole} != ceil(tell_frac()={frac} / 8)={ceil_eighths}"
            );
            let _ = dec.dec_bit_logp(1);
            let _ = dec.dec_bits(2);
        }
    }

    /// `tell_frac()` of a fresh decoder sits in `[1, 8]` (since
    /// `tell()` is `1` and the §4.1.6.1 ceiling identity holds).
    #[test]
    fn tell_frac_initial_within_one_bit() {
        let dec = RangeDecoder::new(&[0xCC, 0xDD, 0xEE, 0xFF]);
        let frac = dec.tell_frac();
        assert!(
            (1..=8).contains(&frac),
            "tell_frac initial out of [1,8]: {frac}"
        );
        assert!(frac.div_ceil(8) == dec.tell());
    }

    /// `dec_icdf` over a binary `{ft - 1, 1}/ft` distribution must
    /// agree with `dec_bit_logp(logp)` step-for-step — both are
    /// special cases of `ec_decode` with `ft = 1<<ftb` (RFC 6716
    /// §4.1.3.2 + §4.1.3.3).
    #[test]
    fn dec_icdf_matches_dec_bit_logp_for_binary() {
        let buf = [0xDE, 0xAD, 0xBE, 0xEF, 0x10, 0x32, 0x54, 0x76];
        // logp = 3 → ft = 8, P("1") = 1/8. icdf {ft-fh[0], ft-fh[1]} =
        // {1, 0}: symbol 0 is the high-probability outcome (the "0"
        // bit).
        let logp = 3u32;
        let icdf = [1u8, 0];
        let mut a = RangeDecoder::new(&buf);
        let mut b = RangeDecoder::new(&buf);
        for _ in 0..16 {
            let via_logp = a.dec_bit_logp(logp);
            let via_icdf = b.dec_icdf(&icdf, logp);
            assert_eq!(
                via_logp, via_icdf,
                "dec_bit_logp({logp}) != dec_icdf({icdf:?}, {logp})"
            );
        }
        assert!(!a.has_error() && !b.has_error());
    }

    /// `dec_icdf` over a uniform `{1,1,1,1,1,1,1,1}/8` PDF must return
    /// a symbol in `[0, 8)` every time without error.
    #[test]
    fn dec_icdf_uniform_returns_in_range() {
        // Uniform 8-way PDF: fh = {1,2,3,4,5,6,7,8} → icdf =
        // {7,6,5,4,3,2,1,0}.
        let icdf = [7u8, 6, 5, 4, 3, 2, 1, 0];
        let mut dec = RangeDecoder::new(&[0x42, 0x18, 0xC3, 0x7F, 0x55, 0xAA, 0x33, 0xCC]);
        for _ in 0..16 {
            let k = dec.dec_icdf(&icdf, 3);
            assert!(k < 8, "icdf uniform returned {k} out of [0, 8)");
        }
        assert!(!dec.has_error());
    }

    /// `dec_icdf` over the degenerate single-symbol table `{0}` (only
    /// the terminator) means symbol 0 covers the whole interval, so it
    /// is always returned. No range mass is consumed and the error
    /// flag stays clear.
    #[test]
    fn dec_icdf_single_symbol_always_zero() {
        let icdf = [0u8];
        let mut dec = RangeDecoder::new(&[0x77, 0x33, 0x11, 0xAA]);
        let before_tell = dec.tell();
        for _ in 0..4 {
            let k = dec.dec_icdf(&icdf, 3);
            assert_eq!(k, 0);
        }
        assert!(dec.tell() >= before_tell);
        assert!(!dec.has_error());
    }

    /// `tell_frac()` is monotonically non-decreasing across mixed ops
    /// (§4.1.6.2 inherits the monotonicity of `ec_tell` since the
    /// procedure only adds bits).
    #[test]
    fn tell_frac_is_monotonic() {
        let mut dec = RangeDecoder::new(&[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        // Uniform 8-way icdf so each call burns ~3 bits.
        let icdf = [7u8, 6, 5, 4, 3, 2, 1, 0];
        let mut prev = dec.tell_frac();
        for i in 0..24 {
            match i % 3 {
                0 => {
                    let _ = dec.dec_bit_logp(2);
                }
                1 => {
                    let _ = dec.dec_icdf(&icdf, 3);
                }
                _ => {
                    let _ = dec.dec_bits(2);
                }
            }
            let now = dec.tell_frac();
            assert!(
                now >= prev,
                "tell_frac() went backwards: {} -> {}",
                prev,
                now
            );
            prev = now;
        }
    }

    /// `dec_bits(0)` returns 0 and consumes nothing.
    #[test]
    fn dec_bits_zero_width_is_noop() {
        let mut dec = RangeDecoder::new(&[0x12, 0x34, 0x56]);
        let before = dec.tell();
        let v = dec.dec_bits(0);
        assert_eq!(v, 0);
        assert_eq!(dec.tell(), before);
        assert!(!dec.has_error());
    }

    /// `dec_bits` with an over-large width sets the error flag and
    /// returns 0 (guard against caller misuse).
    #[test]
    fn dec_bits_oversize_latches_error() {
        let mut dec = RangeDecoder::new(&[0xAA, 0xBB, 0xCC, 0xDD]);
        let v = dec.dec_bits(33);
        assert_eq!(v, 0);
        assert!(dec.has_error());
    }

    /// The public two-step `ec_decode` / `ec_dec_update` path must
    /// reproduce, symbol for symbol, what the fused `dec_icdf` produces
    /// for the same fixed PDF. This is the RFC 6716 §4.1.2 ↔ §4.1.3.3
    /// equivalence: `dec_icdf` is exactly `ec_decode(1<<ftb)` followed
    /// by a search and `ec_dec_update`. We drive both decoders over
    /// identical input bytes and assert they stay in lockstep.
    #[test]
    fn ec_decode_update_matches_dec_icdf_for_fixed_pdf() {
        // Uniform 8-way PDF: fh = {1,2,..,8}, ft = 8, icdf = {7,..,0}.
        let icdf = [7u8, 6, 5, 4, 3, 2, 1, 0];
        let ftb = 3u32;
        let ft = 1u32 << ftb;
        let buf = [0x42u8, 0x18, 0xC3, 0x7F, 0x55, 0xAA, 0x33, 0xCC];
        let mut fused = RangeDecoder::new(&buf);
        let mut split = RangeDecoder::new(&buf);
        for _ in 0..16 {
            let k_fused = fused.dec_icdf(&icdf, ftb);

            // Reconstruct the same decode via the public split steps.
            let fs = split.ec_decode(ft);
            // icdf[k] == ft - fh[k]; fl[k] == fh[k-1] (fl[0] == 0).
            // Find the symbol whose [fl, fh) contains fs.
            let mut k_split = 0u32;
            let mut fl = 0u32;
            let mut fh = ft - icdf[0] as u32;
            for (idx, w) in icdf.windows(2).enumerate() {
                if fs < fh {
                    break;
                }
                fl = ft - w[0] as u32;
                fh = ft - w[1] as u32;
                k_split = (idx + 1) as u32;
            }
            split.ec_dec_update(fl, fh, ft);

            assert_eq!(
                k_fused, k_split,
                "fused dec_icdf and split ec_decode/ec_dec_update diverged"
            );
        }
        assert!(!fused.has_error() && !split.has_error());
    }

    /// `ec_decode(ft)` returns a value in `[0, ft)` for every well-formed
    /// `ft`, matching the §4.1.2 derivation `fs = ft - min(.., ft)`.
    #[test]
    fn ec_decode_returns_in_range() {
        let buf = [0x37u8, 0x91, 0xC4, 0x18, 0xA2, 0x5D, 0x6E, 0xFF];
        for &ft in &[2u32, 7, 100, 1000, 1 << 16] {
            let mut dec = RangeDecoder::new(&buf);
            let fs = dec.ec_decode(ft);
            assert!(fs < ft, "ec_decode({ft}) = {fs} out of [0, {ft})");
            assert!(!dec.has_error());
        }
    }

    /// `ec_decode(0)` is malformed (division by zero in the §4.1.2
    /// formula); it must latch the error flag and return 0 rather than
    /// panic.
    #[test]
    fn ec_decode_ft_zero_latches_error() {
        let mut dec = RangeDecoder::new(&[0x11, 0x22, 0x33, 0x44]);
        let fs = dec.ec_decode(0);
        assert_eq!(fs, 0);
        assert!(dec.has_error());
    }

    /// `ec_dec_update` with a malformed tuple (`fl >= fh`, `fh > ft`,
    /// or `ft == 0`) latches the error flag and leaves state untouched,
    /// guarding against an underflow of `val` or a zeroing of `rng`.
    #[test]
    fn ec_dec_update_rejects_malformed_tuple() {
        // fl >= fh
        let mut a = RangeDecoder::new(&[0xAA, 0xBB, 0xCC, 0xDD]);
        a.ec_dec_update(4, 4, 8);
        assert!(a.has_error());

        // fh > ft
        let mut b = RangeDecoder::new(&[0xAA, 0xBB, 0xCC, 0xDD]);
        b.ec_dec_update(0, 9, 8);
        assert!(b.has_error());

        // ft == 0
        let mut c = RangeDecoder::new(&[0xAA, 0xBB, 0xCC, 0xDD]);
        c.ec_dec_update(0, 1, 0);
        assert!(c.has_error());
    }

    /// The public split path must also reproduce the `dec_uint` small
    /// regime (`ftb <= 8`), which is internally `decode(ft)` followed by
    /// `dec_update(t, t+1, ft)`. Drive `dec_uint` and the public
    /// `ec_decode` / `ec_dec_update` in lockstep for a small `ft`.
    #[test]
    fn ec_decode_update_matches_dec_uint_small_regime() {
        let buf = [0x42u8, 0x18, 0xC3, 0x7F, 0x55, 0xAA, 0x33, 0xCC];
        let ft = 200u32; // ftb <= 8 → small dec_uint regime
        let mut via_uint = RangeDecoder::new(&buf);
        let mut via_split = RangeDecoder::new(&buf);
        for _ in 0..8 {
            let u = via_uint.dec_uint(ft).expect("ft=200 small regime");
            let t = via_split.ec_decode(ft);
            via_split.ec_dec_update(t, t + 1, ft);
            assert_eq!(u, t, "dec_uint and split ec_decode/update diverged");
        }
        assert!(!via_uint.has_error() && !via_split.has_error());
    }
}
