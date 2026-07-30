//! Range encoder primitives for the Opus codec.
//!
//! This module implements the bit-exact range *encoder* described in
//! RFC 6716 §5.1 (`docs/audio/opus/rfc6716-opus.txt`). It is the exact
//! counterpart of the [`crate::range_decoder`] module: any sequence of
//! symbols written here decodes back to the identical sequence through
//! [`crate::range_decoder::RangeDecoder`], and — per §5.1 — the encoder
//! `rng` after a symbol matches the decoder `rng` after decoding the
//! same symbol. The implementation is clean-room: every routine is
//! transcribed from the prose and equations in RFC 6716 §5.1; no
//! external library source was consulted.
//!
//! The range encoder is the SHARED entropy primitive that both the SILK
//! and CELT layers of an Opus encoder invoke for every coded symbol.
//!
//! The following routines are wired up:
//!
//! * Initialization (§5.1).
//! * [`RangeEncoder::encode`] — the generic `ec_encode(fl, fh, ft)`
//!   symbol-update path (§5.1.1), with renormalization (§5.1.1.1) and
//!   carry propagation / output buffering (§5.1.1.2).
//! * [`RangeEncoder::encode_bin`] for power-of-two `ft = 1<<ftb`
//!   (§5.1.2.1, `ec_encode_bin`).
//! * [`RangeEncoder::enc_bit_logp`] for a single binary symbol with
//!   probability `2**-logp` of a "1" (§5.1.2.2).
//! * [`RangeEncoder::enc_icdf`] for inverse-CDF table encoding, sharing
//!   the decoder's `icdf[]` tables verbatim (§5.1.2.3).
//! * [`RangeEncoder::enc_bits`] for raw bits packed at the end of the
//!   buffer (§5.1.3).
//! * [`RangeEncoder::enc_uint`] for uniformly-distributed integers
//!   (§5.1.4).
//! * [`RangeEncoder::tell`] / [`RangeEncoder::tell_frac`] for
//!   whole-bit / 1/8th-bit accounting (§5.1.6), matching the decoder's
//!   [`crate::range_decoder::RangeDecoder::tell`] value bit-for-bit
//!   after the same symbols.
//! * [`RangeEncoder::finish`] — stream finalization (§5.1.5,
//!   `ec_enc_done`), which selects the terminating code value and lays
//!   out the range bytes and the trailing raw-bit region.

/// Bit-exact CELT/SILK range encoder state per RFC 6716 §5.1.
///
/// The state four-tuple `(val, rng, rem, ext)` from §5.1 is carried
/// directly: `val` is the low end of the current range, `rng` its size,
/// `rem` a single buffered non-propagating output byte (or `-1` for
/// "none yet"), and `ext` a count of pending carry-propagating (`255`)
/// output bytes. Range-coder bytes accumulate front-to-back in `buf`;
/// raw bits (§5.1.3) accumulate back-to-front in `end_bytes` /
/// `end_window` and are appended as the buffer tail at [`Self::finish`].
#[derive(Debug, Clone)]
pub struct RangeEncoder {
    /// Range-coder output bytes, in forward order (index 0 first).
    buf: Vec<u8>,
    /// Range size; the renormalization invariant is `rng > 2**23`.
    rng: u32,
    /// Low end of the current range (masked to 31 bits at rest).
    val: u32,
    /// Buffered non-propagating output byte, `0..=254`, or `-1` for
    /// "no byte buffered yet" (§5.1.1.2).
    rem: i32,
    /// Count of pending carry-propagating (`255`) output bytes
    /// (§5.1.1.2).
    ext: u32,
    /// Partial raw-bit window: the next bit to emit sits in bit 0
    /// (§5.1.3). Holds fewer than 8 bits at rest.
    end_window: u32,
    /// Number of valid bits currently in `end_window` (0..=7 at rest).
    nend_bits: u32,
    /// Completed raw-bit bytes. `end_bytes[0]` is the LAST byte of the
    /// finished stream, `end_bytes[1]` the second-to-last, and so on —
    /// matching the decoder's back-to-front raw-bit reader (§4.1.4).
    end_bytes: Vec<u8>,
    /// Running tally of whole bits the range coder has produced
    /// (RFC 6716 §5.1.6 / §4.1.6 `nbits_total`).
    nbits_total: u32,
    /// Number of raw bits emitted so far, added into the bit-usage
    /// accounting on top of `nbits_total` (§4.1.6).
    nbits_raw: u32,
}

impl Default for RangeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeEncoder {
    /// Renormalization threshold from §5.1.1.1: normalize until
    /// `rng > 2**23`.
    const RNG_MIN: u32 = 1 << 23;

    /// Depth of the decoder's forward-read lookahead, in bytes. The
    /// 31-bit range window (`val < 2**31`) plus the §4.1.1 initialization
    /// pre-read means the decoder can consume up to one full range window
    /// (4 bytes) beyond the last committed range byte. Used at
    /// [`Self::finish`] to size the zero pad that isolates the raw-bit
    /// tail from the range reader.
    const RANGE_LOOKAHEAD_BYTES: usize = 4;

    /// Initialize the range encoder per RFC 6716 §5.1.
    ///
    /// The state vector is `(val, rng, rem, ext) = (0, 2**31, -1, 0)`.
    /// `nbits_total` starts at 33 so that [`Self::tell`] reports the
    /// same value as a freshly-initialized decoder (which reaches
    /// `nbits_total == 33`, `rng == 2**31`, `tell() == 1` after its
    /// §4.1.1 initialization normalize).
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            rng: 1 << 31,
            val: 0,
            rem: -1,
            ext: 0,
            end_window: 0,
            nend_bits: 0,
            end_bytes: Vec::new(),
            // §4.1.6/§5.1.6: matches the decoder's post-init value so
            // `tell()` agrees symbol-for-symbol.
            nbits_total: 33,
            nbits_raw: 0,
        }
    }

    /// Whole-bit budget produced so far, per RFC 6716 §5.1.6 / §4.1.6.1.
    ///
    /// Equal to `nbits_total - ilog(rng) + nbits_raw`; matches the
    /// decoder's `tell()` after the same symbols.
    pub fn tell(&self) -> u32 {
        let lg = 32 - self.rng.leading_zeros();
        self.nbits_total
            .saturating_sub(lg)
            .saturating_add(self.nbits_raw)
    }

    /// 1/8th-bit-precision budget produced so far, per RFC 6716 §5.1.6 /
    /// §4.1.6.2. Matches the decoder's `tell_frac()` after the same
    /// symbols.
    pub fn tell_frac(&self) -> u32 {
        let lg0 = 32 - self.rng.leading_zeros();
        let mut r_q15 = self.rng >> (lg0 - 16);
        let mut lg_frac = lg0;
        for _ in 0..3 {
            r_q15 = (r_q15 * r_q15) >> 15;
            let bit = r_q15 >> 16;
            lg_frac = 2 * lg_frac + bit;
            if bit == 1 {
                r_q15 >>= 1;
            }
        }
        self.nbits_total
            .saturating_mul(8)
            .saturating_sub(lg_frac)
            .saturating_add(self.nbits_raw.saturating_mul(8))
    }

    /// Encode symbol `k` described by the three-tuple `(fl, fh, ft)`,
    /// per RFC 6716 §5.1.1 (`ec_encode`).
    ///
    /// Requires `0 <= fl < fh <= ft` and `1 <= ft <= 2**16`. The §5.1.1
    /// update narrows the range to the symbol's `[fl, fh)` sub-interval
    /// of `[0, ft)`. The `fl == 0` branch subtracts `(rng/ft)*(ft - fh)`
    /// from `rng` so the top symbol absorbs the integer-division
    /// remainder — the exact mirror of the decoder's §4.1.2 update, which
    /// is what keeps encoder `rng` equal to decoder `rng`.
    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        debug_assert!(fl < fh && fh <= ft && ft >= 1);
        let r = self.rng / ft;
        if fl > 0 {
            self.val = self.val.wrapping_add(self.rng - r.wrapping_mul(ft - fl));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng -= r.wrapping_mul(ft - fh);
        }
        self.normalize();
    }

    /// Encode symbol `k` for a power-of-two total `ft = 1<<ftb`, per
    /// RFC 6716 §5.1.2.1 (`ec_encode_bin`). Division-free equivalent of
    /// [`Self::encode`] with `ft = 1<<ftb`.
    pub fn encode_bin(&mut self, fl: u32, fh: u32, ftb: u32) {
        let ft = 1u32 << ftb;
        debug_assert!(fl < fh && fh <= ft);
        let r = self.rng >> ftb;
        if fl > 0 {
            self.val = self.val.wrapping_add(self.rng - r.wrapping_mul(ft - fl));
            self.rng = r.wrapping_mul(fh - fl);
        } else {
            self.rng -= r.wrapping_mul(ft - fh);
        }
        self.normalize();
    }

    /// Encode a single binary symbol whose "1" has probability
    /// `2**-logp`, per RFC 6716 §5.1.2.2 (`ec_enc_bit_logp`).
    ///
    /// Equivalent to `ec_encode` with `(fl, fh, ft)` equal to
    /// `(0, (1<<logp)-1, 1<<logp)` for a `0` and
    /// `((1<<logp)-1, 1<<logp, 1<<logp)` for a `1`. Multiplication- and
    /// division-free.
    pub fn enc_bit_logp(&mut self, bit: bool, logp: u32) {
        let r = self.rng >> logp;
        if bit {
            // fl = (1<<logp)-1, fh = ft = 1<<logp; fh-fl = 1, ft-fl = 1.
            self.val = self.val.wrapping_add(self.rng - r);
            self.rng = r;
        } else {
            // fl = 0, fh = (1<<logp)-1, ft = 1<<logp; ft-fh = 1.
            self.rng -= r;
        }
        self.normalize();
    }

    /// Encode symbol index `k` against an inverse-CDF table, per
    /// RFC 6716 §5.1.2.3 (`ec_enc_icdf`). Uses the SAME `icdf[]` tables
    /// as the decoder's [`crate::range_decoder::RangeDecoder::dec_icdf`]:
    /// `icdf[j]` stores `(1<<ftb) - fh[j]`, terminated by a `0` entry.
    ///
    /// Per §5.1.2.3, `fl[k] = (1<<ftb) - icdf[k-1]` (or `0` if `k == 0`),
    /// `fh[k] = (1<<ftb) - icdf[k]`, `ft = 1<<ftb`. The symbol update
    /// then depends only on the `icdf[]` differences, so no total is
    /// needed.
    pub fn enc_icdf(&mut self, k: usize, icdf: &[u8], ftb: u32) {
        // A zero-width symbol (fh == fl: a zero-probability PDF cell)
        // would collapse `rng` to zero and make the stream undecodable;
        // callers must never encode one.
        debug_assert!(
            if k == 0 {
                (icdf[0] as u32) < (1u32 << ftb)
            } else {
                icdf[k - 1] > icdf[k]
            },
            "zero-probability icdf cell {k}"
        );
        let r = self.rng >> ftb;
        if k > 0 {
            let hi = icdf[k - 1] as u32;
            let lo = icdf[k] as u32;
            // ft - fl = icdf[k-1]; fh - fl = icdf[k-1] - icdf[k].
            self.val = self.val.wrapping_add(self.rng - r.wrapping_mul(hi));
            self.rng = r.wrapping_mul(hi - lo);
        } else {
            // fl = 0; ft - fh = icdf[0].
            self.rng -= r.wrapping_mul(icdf[0] as u32);
        }
        self.normalize();
    }

    /// Encode `bits` raw bits (low bits of `value`), per RFC 6716
    /// §5.1.3 (`ec_enc_bits`). Raw bits are packed at the END of the
    /// output buffer, LSB-first, mirroring the decoder's back-to-front
    /// reader. The first bit emitted here is the one the decoder reads
    /// first.
    pub fn enc_bits(&mut self, value: u32, bits: u32) {
        debug_assert!(bits <= 32);
        if bits == 0 {
            return;
        }
        let mask: u64 = if bits >= 32 {
            0xFFFF_FFFF
        } else {
            (1u64 << bits) - 1
        };
        let mut window = (self.end_window as u64) | ((value as u64 & mask) << self.nend_bits);
        let mut n = self.nend_bits + bits;
        while n >= 8 {
            self.end_bytes.push((window & 0xFF) as u8);
            window >>= 8;
            n -= 8;
        }
        self.end_window = window as u32;
        self.nend_bits = n;
        self.nbits_raw += bits;
    }

    /// Encode one of `ft` equiprobable values `t` in `0..ft`, per
    /// RFC 6716 §5.1.4 (`ec_enc_uint`). `ft` may be as large as
    /// `2**32 - 1`. Values `ft <= 1` degenerate to a no-op (matching the
    /// decoder returning the constant `0`).
    pub fn enc_uint(&mut self, t: u32, ft: u32) {
        debug_assert!(ft >= 1 && t < ft.max(1));
        if ft <= 1 {
            return;
        }
        // ftb = ilog(ft - 1): bits needed to store ft-1.
        let ftb = 32 - (ft - 1).leading_zeros();
        if ftb <= 8 {
            self.encode(t, t + 1, ft);
        } else {
            let split = ftb - 8;
            let hi = t >> split;
            let top = ((ft - 1) >> split) + 1;
            self.encode(hi, hi + 1, top);
            self.enc_bits(t & ((1u32 << split) - 1), split);
        }
    }

    /// The current range size `rng` (RFC 6716 §5.1.1). After the same
    /// symbol sequence this equals the decoder's `rng`
    /// ([`crate::range_decoder::RangeDecoder::range_size`]); the value
    /// at end-of-frame is the "final range" diagnostic both sides can
    /// compare.
    #[must_use]
    pub fn range_size(&self) -> u32 {
        self.rng
    }

    /// Finalize the stream into a **fixed-size** buffer of exactly
    /// `size` bytes (§5.1.5, `ec_enc_done` with a bounded `storage`):
    /// range-coder bytes grow from the front, §5.1.3 raw bits from the
    /// back, and the unused middle is zero — the layout a CELT frame
    /// requires, where the frame's byte size *is* the entropy budget
    /// and the §4.1.2 decoder reads the same buffer from both ends.
    ///
    /// The final partial raw-bit window is OR-merged into the last raw
    /// byte exactly as the reference listing's finalization does; when
    /// the range data and the raw data meet in one byte, the range
    /// value's trailing free bits absorb the raw bits.
    ///
    /// Returns `None` when the coded symbols cannot fit `size` bytes
    /// (an encoder-side budget-accounting bug: every CELT symbol is
    /// gated on `tell()` against the same `size * 8` budget).
    #[must_use]
    pub fn finish_fixed(mut self, size: usize) -> Option<Vec<u8>> {
        // Minimum number of terminating bits so the symbols decode
        // correctly regardless of what follows.
        let mut l: i32 = 32 - (32 - self.rng.leading_zeros()) as i32;
        let mut msk: u32 = 0x7FFF_FFFFu32 >> l;
        let mut end: u32 = self.val.wrapping_add(msk) & !msk;
        if u64::from(end | msk) >= u64::from(self.val) + u64::from(self.rng) {
            l += 1;
            msk >>= 1;
            end = self.val.wrapping_add(msk) & !msk;
        }
        while l > 0 {
            self.carry_out(end >> 23);
            end = (end << 8) & 0x7FFF_FFFF;
            l -= 8;
        }
        // Flush the buffered byte / pending carry run.
        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }

        let n_range = self.buf.len();
        let n_raw = self.end_bytes.len();
        if n_range + n_raw > size {
            return None;
        }
        let mut out = vec![0u8; size];
        out[..n_range].copy_from_slice(&self.buf);
        for (i, &b) in self.end_bytes.iter().enumerate() {
            out[size - 1 - i] = b;
        }
        // Remaining partial raw-bit window: OR into the next raw byte.
        if self.nend_bits > 0 {
            if n_raw >= size {
                return None;
            }
            let mut window = self.end_window;
            // If the raw bits share the last range byte, only the
            // range value's trailing free bits (`-l`) are usable.
            if n_range + n_raw >= size && (-l as u32) < self.nend_bits {
                return None;
            }
            if n_range + n_raw >= size {
                window &= (1u32 << (-l)) - 1;
            }
            out[size - n_raw - 1] |= window as u8;
        }
        Some(out)
    }

    /// Finalize the stream (§5.1.5, `ec_enc_done`) and return the packed
    /// output bytes.
    ///
    /// Chooses the terminating code value `end` inside `[val, val + rng)`
    /// with the most trailing zero bits `b` such that
    /// `end + (1<<b) - 1` is still in the interval, flushes it through
    /// the carry buffer, then appends the raw-bit region (§5.1.3) as a
    /// disjoint tail, separated from the range data by a zero pad so the
    /// decoder's forward range reader can zero-extend past the committed
    /// bytes without consuming a raw byte.
    pub fn finish(mut self) -> Vec<u8> {
        let val = self.val;
        let rng = self.rng;
        // Default b = 0: end = val is always in the interval.
        let mut end = val;
        // Pick the largest b in 1..=31 whose rounded-up multiple of 2**b
        // keeps `end + (2**b - 1)` inside `[val, val + rng)`.
        for b in (1..=31u32).rev() {
            let m = (1u64 << b) - 1;
            let end_b = ((val as u64) + m) & !m;
            if end_b + m < (val as u64) + (rng as u64) {
                end = end_b as u32;
                break;
            }
        }
        // Flush `end` through the carry buffer, 9 bits (top of `end`) at
        // a time.
        while end != 0 {
            self.carry_out(end >> 23);
            end = (end << 8) & 0x7FFF_FFFF;
        }
        // Flush the buffered byte to the output (§5.1.5): if `rem` holds
        // a real non-zero byte, or a carry run is pending, emit 9 zero
        // bits.
        if (self.rem != -1 && self.rem != 0) || self.ext > 0 {
            self.carry_out(0);
        }
        // Append the raw bits (§5.1.3) as a disjoint tail. The §5.1.5
        // `end` finalization commits the range value to the front bytes
        // and relies on the decoder reading TRAILING ZEROS beyond them
        // (the chosen `end` maximizes trailing zero bits). The decoder's
        // forward range reader runs up to `RANGE_LOOKAHEAD_BYTES` bytes
        // ahead of the committed data — its §4.1.1 initialization alone
        // pre-reads a full range window before any symbol — so those
        // lookahead positions MUST read as zero. When raw bits are
        // present they occupy the buffer tail; a zero pad of one full
        // range window separates them from the range data so the range
        // reader's lookahead never consumes a raw byte. The raw region
        // is laid out so the decoder's back-to-front raw reader (§4.1.4)
        // sees `end_bytes[0]` (the first-emitted 8 raw bits) as the very
        // last byte, then earlier full bytes, then the partial window
        // byte (the last-emitted, fewer-than-8 bits) closest to the pad.
        let mut out = self.buf;
        let have_raw = self.nend_bits > 0 || !self.end_bytes.is_empty();
        if have_raw {
            out.resize(out.len() + Self::RANGE_LOOKAHEAD_BYTES, 0);
            if self.nend_bits > 0 {
                out.push(self.end_window as u8);
            }
            for &b in self.end_bytes.iter().rev() {
                out.push(b);
            }
        }
        out
    }

    // ----- internal helpers -----

    /// `ec_enc_normalize` per RFC 6716 §5.1.1.1: while `rng <= 2**23`,
    /// spill the top 9 bits of `val` to the carry buffer and shift both
    /// `val` and `rng` left by 8.
    fn normalize(&mut self) {
        while self.rng <= Self::RNG_MIN {
            self.carry_out(self.val >> 23);
            self.val = (self.val << 8) & 0x7FFF_FFFF;
            self.rng <<= 8;
            self.nbits_total = self.nbits_total.saturating_add(8);
        }
    }

    /// `ec_enc_carry_out` per RFC 6716 §5.1.1.2. Takes a 9-bit value
    /// `c` (8 data bits + 1 carry bit).
    fn carry_out(&mut self, c: u32) {
        if c == 0xFF {
            // All-ones data with no carry: defer as a potential carry
            // run.
            self.ext += 1;
            return;
        }
        let b = c >> 8; // carry bit, 0 or 1
        if self.rem >= 0 {
            self.buf.push((self.rem as u32 + b) as u8);
        }
        if self.ext > 0 {
            // Resolve the deferred 255-run: 0x00 if the carry
            // propagates, 0xFF otherwise.
            let fill = if b != 0 { 0x00 } else { 0xFF };
            for _ in 0..self.ext {
                self.buf.push(fill);
            }
            self.ext = 0;
        }
        self.rem = (c & 0xFF) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_decoder::RangeDecoder;

    /// A tiny deterministic PRNG so the fuzz roundtrips need no external
    /// crate. Not cryptographic; only used to drive symbol choices.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            // Numerical Recipes LCG constants.
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 32) as u32
        }
        fn below(&mut self, n: u32) -> u32 {
            if n == 0 {
                0
            } else {
                self.next_u32() % n
            }
        }
    }

    /// §5.1: a freshly-initialized encoder reports the same `tell()` as a
    /// freshly-initialized decoder: 1 bit.
    #[test]
    fn finish_fixed_shares_one_buffer_between_range_and_raw_bits() {
        // CELT layout: range symbols from the front, raw bits from the
        // back, in EXACTLY `size` bytes. Everything must decode back
        // from that fixed buffer.
        for size in [6usize, 12, 40] {
            let mut enc = RangeEncoder::new();
            let mut lcg = Lcg(size as u64 * 977 + 5);
            let mut icdf_syms = Vec::new();
            let mut bit_syms = Vec::new();
            let mut raw_syms = Vec::new();
            const ICDF: [u8; 4] = [25, 23, 2, 0];
            // Fill roughly half the budget with range symbols, a
            // quarter with raw bits.
            while enc.tell() + 16 < (size as u32 * 8) / 2 {
                let k = lcg.below(4) as usize;
                enc.enc_icdf(k, &ICDF, 5);
                icdf_syms.push(k);
                let b = lcg.below(2) == 1;
                enc.enc_bit_logp(b, 3);
                bit_syms.push(b);
            }
            for _ in 0..size / 4 {
                let v = lcg.below(16);
                enc.enc_bits(v, 4);
                raw_syms.push(v);
            }
            assert!(enc.tell() <= size as u32 * 8, "over budget");
            let buf = enc.finish_fixed(size).expect("fits");
            assert_eq!(buf.len(), size);
            let mut rd = RangeDecoder::new(&buf);
            for (i, &k) in icdf_syms.iter().enumerate() {
                assert_eq!(rd.dec_icdf(&ICDF, 5) as usize, k, "size {size} icdf #{i}");
                assert_eq!(rd.dec_bit_logp(3) == 1, bit_syms[i], "size {size} bit #{i}");
            }
            for (i, &v) in raw_syms.iter().enumerate() {
                assert_eq!(rd.dec_bits(4), v, "size {size} raw #{i}");
            }
            assert!(!rd.has_error());
        }
    }

    #[test]
    fn init_tell_is_one() {
        let enc = RangeEncoder::new();
        let dec = RangeDecoder::new(&[]);
        assert_eq!(enc.tell(), 1);
        assert_eq!(enc.tell(), dec.tell());
        // tell_frac reports 1/8th bits: 1 whole bit == 8 eighths.
        assert_eq!(enc.tell_frac(), 8);
        assert_eq!(enc.tell_frac(), dec.tell_frac());
    }

    /// §5.1.2.3 / §4.1.3.3: encode a sequence of icdf symbols and decode
    /// them back — the decoded indices must match exactly.
    #[test]
    fn roundtrip_icdf() {
        // A valid inverse-CDF table (strictly decreasing, terminated by
        // 0) with ftb = 8, i.e. ft = 256.
        let icdf: [u8; 4] = [200, 120, 40, 0];
        let ftb = 8;
        let symbols = [0usize, 3, 1, 2, 2, 0, 1, 3, 3, 0, 2, 1];
        let mut enc = RangeEncoder::new();
        for &k in &symbols {
            enc.enc_icdf(k, &icdf, ftb);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for &k in &symbols {
            assert_eq!(dec.dec_icdf(&icdf, ftb) as usize, k);
        }
        assert!(!dec.has_error());
    }

    /// §5.1.2.2 / §4.1.3.2: bit_logp roundtrip.
    #[test]
    fn roundtrip_bit_logp() {
        let bits = [
            true, false, false, true, true, true, false, true, false, false,
        ];
        let logps = [1u32, 2, 4, 8, 3, 6, 2, 1, 5, 7];
        let mut enc = RangeEncoder::new();
        for (i, &bit) in bits.iter().enumerate() {
            enc.enc_bit_logp(bit, logps[i]);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for (i, &bit) in bits.iter().enumerate() {
            assert_eq!(dec.dec_bit_logp(logps[i]) == 1, bit);
        }
        assert!(!dec.has_error());
    }

    /// §5.1.3 / §4.1.4: full-width 32-bit raw reads and writes are
    /// defined (a round-382 fuzz find: the decoder's u32 raw-bit window
    /// overflowed its shifts on a 32-bit read).
    #[test]
    fn roundtrip_raw_bits_32_wide() {
        let vals: [u32; 5] = [0xFFFF_FFFF, 0, 0x8000_0001, 0xDEAD_BEEF, 0x0BAD_F00D];
        let mut enc = RangeEncoder::new();
        // A 3-bit write first so the 32-bit reads straddle byte seams.
        enc.enc_bits(0b101, 3);
        for &v in &vals {
            enc.enc_bits(v, 32);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        assert_eq!(dec.dec_bits(3), 0b101);
        for &v in &vals {
            assert_eq!(dec.dec_bits(32), v);
        }
        assert!(!dec.has_error());
    }

    /// §5.1.3 / §4.1.4: raw-bit roundtrip, mixed widths.
    #[test]
    fn roundtrip_raw_bits() {
        let vals: [(u32, u32); 6] = [
            (1, 1),
            (0b101, 3),
            (0xAB, 8),
            (0x1234, 16),
            (0, 4),
            (0x7F, 7),
        ];
        let mut enc = RangeEncoder::new();
        for &(v, b) in &vals {
            enc.enc_bits(v, b);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for &(v, b) in &vals {
            assert_eq!(dec.dec_bits(b), v);
        }
        assert!(!dec.has_error());
    }

    /// §5.1.4 / §4.1.5: uint roundtrip across the small (ftb<=8) and
    /// large (ftb>8, raw-bit tail) paths, interleaved.
    #[test]
    fn roundtrip_uint() {
        let cases: [(u32, u32); 8] = [
            (0, 1),
            (3, 4),
            (200, 256),
            (1000, 1024),
            (65535, 65536),
            (7, 100),
            (123456, 1_000_000),
            (0, 300),
        ];
        let mut enc = RangeEncoder::new();
        for &(t, ft) in &cases {
            enc.enc_uint(t, ft);
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for &(t, ft) in &cases {
            assert_eq!(dec.dec_uint(ft).unwrap(), t);
        }
        assert!(!dec.has_error());
    }

    /// §5.1.1: generic ec_encode roundtrip using a small uniform model,
    /// decoded via the split ec_decode / ec_dec_update path. Encoder and
    /// decoder `rng` are compared symbol-for-symbol via the §5.1.6
    /// `tell()` cross-check (`tell` is a pure function of `rng` and the
    /// symbol-count-driven `nbits_total`, which both sides track
    /// identically).
    #[test]
    fn roundtrip_ec_encode_uniform() {
        let ft = 11u32; // arbitrary non-power-of-two
        let symbols = [0u32, 10, 5, 3, 7, 1, 9, 2, 8, 4, 6, 0, 10, 5];
        let mut enc = RangeEncoder::new();
        let mut enc_tell = Vec::new();
        for &k in &symbols {
            enc.encode(k, k + 1, ft);
            enc_tell.push(enc.tell());
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for (i, &k) in symbols.iter().enumerate() {
            let fs = dec.ec_decode(ft);
            // Uniform model: fs directly identifies the symbol here.
            assert_eq!(fs, k);
            dec.ec_dec_update(fs, fs + 1, ft);
            assert_eq!(dec.tell(), enc_tell[i], "tell (rng) desync at symbol {i}");
        }
        assert!(!dec.has_error());
    }

    /// §5.1.6: `tell()` / `tell_frac()` track the decoder bit-for-bit.
    #[test]
    fn tell_matches_decoder() {
        let icdf: [u8; 3] = [170, 50, 0];
        let symbols = [0usize, 2, 1, 1, 0, 2, 2, 1, 0];
        let mut enc = RangeEncoder::new();
        let mut enc_tell = Vec::new();
        let mut enc_tell_frac = Vec::new();
        for &k in &symbols {
            enc.enc_icdf(k, &icdf, 8);
            enc_tell.push(enc.tell());
            enc_tell_frac.push(enc.tell_frac());
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes);
        for (i, &k) in symbols.iter().enumerate() {
            assert_eq!(dec.dec_icdf(&icdf, 8) as usize, k);
            assert_eq!(dec.tell(), enc_tell[i], "tell desync at {i}");
            assert_eq!(dec.tell_frac(), enc_tell_frac[i], "tell_frac desync at {i}");
        }
    }

    /// Randomized fuzz: mix icdf / bit_logp / uint / raw-bit symbols and
    /// require the decoder to recover every one across many seeds.
    #[test]
    fn fuzz_mixed_roundtrip() {
        let icdf: [u8; 5] = [220, 150, 90, 30, 0];
        for seed in 0..5000u64 {
            let mut rng = Lcg(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1));
            let count = 8 + rng.below(60);
            let mut ops: Vec<u8> = Vec::new();
            let mut icdf_syms: Vec<usize> = Vec::new();
            let mut logp_bits: Vec<(bool, u32)> = Vec::new();
            let mut uint_vals: Vec<(u32, u32)> = Vec::new();
            let mut raw_vals: Vec<(u32, u32)> = Vec::new();

            let mut enc = RangeEncoder::new();
            for _ in 0..count {
                match rng.below(4) {
                    0 => {
                        let k = rng.below(4) as usize; // 0..=3 valid symbols
                        enc.enc_icdf(k, &icdf, 8);
                        ops.push(0);
                        icdf_syms.push(k);
                    }
                    1 => {
                        let bit = rng.below(2) == 1;
                        let logp = 1 + rng.below(8);
                        enc.enc_bit_logp(bit, logp);
                        ops.push(1);
                        logp_bits.push((bit, logp));
                    }
                    2 => {
                        let ft = 2 + rng.below(1 << 20);
                        let t = rng.below(ft);
                        enc.enc_uint(t, ft);
                        ops.push(2);
                        uint_vals.push((t, ft));
                    }
                    _ => {
                        let b = 1 + rng.below(16);
                        let v = rng.next_u32() & if b >= 32 { !0 } else { (1u32 << b) - 1 };
                        enc.enc_bits(v, b);
                        ops.push(3);
                        raw_vals.push((v, b));
                    }
                }
            }
            let bytes = enc.finish();
            let mut dec = RangeDecoder::new(&bytes);
            let (mut ii, mut li, mut ui, mut ri) = (0usize, 0usize, 0usize, 0usize);
            for &op in &ops {
                match op {
                    0 => {
                        assert_eq!(
                            dec.dec_icdf(&icdf, 8) as usize,
                            icdf_syms[ii],
                            "seed {seed}"
                        );
                        ii += 1;
                    }
                    1 => {
                        let (bit, logp) = logp_bits[li];
                        assert_eq!(dec.dec_bit_logp(logp) == 1, bit, "seed {seed}");
                        li += 1;
                    }
                    2 => {
                        let (t, ft) = uint_vals[ui];
                        assert_eq!(dec.dec_uint(ft).unwrap(), t, "seed {seed}");
                        ui += 1;
                    }
                    _ => {
                        let (v, b) = raw_vals[ri];
                        assert_eq!(dec.dec_bits(b), v, "seed {seed}");
                        ri += 1;
                    }
                }
            }
            assert!(!dec.has_error(), "seed {seed} latched error");
        }
    }
}
