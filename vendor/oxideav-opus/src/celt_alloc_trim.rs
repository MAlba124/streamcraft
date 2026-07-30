//! CELT §4.3.3 allocation-trim parameter surface (RFC 6716 §4.3.3,
//! pp. 114–115; Table 58 on p. 115).
//!
//! The §4.3.3 *Bit Allocation* procedure decodes a single integer
//! `alloc_trim ∈ 0..=10` that biases the Table 57 static allocation
//! toward lower or higher MDCT bands. The §4.3.3 narrative reads
//! (RFC 6716 §4.3.3, p. 114):
//!
//! > The allocation trim is an integer value from 0-10. The default
//! > value of 5 indicates no trim. The trim parameter is entropy
//! > coded in order to lower the coding cost of less extreme
//! > adjustments. Values lower than 5 bias the allocation towards
//! > lower frequencies and values above 5 bias it towards higher
//! > frequencies. Like other signaled parameters, signaling of the
//! > trim is gated so that it is not included if there is
//! > insufficient space available in the bitstream. To decode the
//! > trim, first set the trim value to 5, then if and only if the
//! > count of decoded 8th bits so far (ec_tell_frac) plus 48 (6 bits)
//! > is less than or equal to the total frame size in 8th bits minus
//! > total_boost (a product of the above band boost procedure),
//! > decode the trim value using the PDF in Table 58.
//!
//! Table 58 (RFC 6716 §4.3.3, p. 115) is the 11-cell PDF
//! `{2, 2, 5, 10, 22, 46, 22, 10, 5, 2, 2}/128`. Symbol `k ∈ 0..=10`
//! reads as the trim integer `k`. The PDF is symmetric around `k = 5`
//! (the default) with the heaviest mass on the no-trim cell; the
//! shape matches the §4.3.3 narrative's "less extreme adjustments
//! cheapened" statement.
//!
//! This module owns only the §4.3.3 *parameter surface*: the PDF /
//! iCDF reproduced inline, the §4.3.3 gate predicate
//! `(ec_tell_frac + 48) ≤ (frame_size_bytes * 8 − total_boost)`, the
//! `AllocTrim::DEFAULT = 5` and `AllocTrim::{MIN, MAX} = 0..=10`
//! constants, and the typed decode wrapper that fuses the gate with
//! the [`crate::range_decoder::RangeDecoder::dec_icdf`] call. The
//! §4.3.3 *use* of the trim — the per-band trim_offsets[] derivation
//! and the consequent shift in the Table 57 static allocation search
//! — is the responsibility of the §4.3.3 allocator and runs at the
//! call site.
//!
//! ## §4.3.3 signalling gate
//!
//! Per RFC 6716 §4.3.3 (p. 114), the trim is signalled iff
//!
//! ```text
//!     ec_tell_frac() + 48 ≤ (frame_size_bytes * 8) − total_boost
//! ```
//!
//! where the units are 1/8 bits ("8th bits") throughout. The `48`
//! is the conservative budget for the worst-case trim symbol: six
//! whole bits × 8 = 48 1/8 bits. `total_boost` is the §4.3.3 band
//! boost accumulator from the §4.3.3 boost loop (already in 1/8
//! bits). When the gate fails, the trim is left at its default
//! `5` (no trim) and no bits are consumed.
//!
//! ## Provenance
//!
//! Narrative: RFC 6716 §4.3.3 (pp. 114–115) in
//! `docs/audio/opus/rfc6716-opus.txt`; cross-referenced by
//! `docs/audio/celt/spec/celt-coarse-energy-and-allocation.md` §2.4.
//! Table 58 PDF is inlined in the RFC body (p. 115) — no separate
//! CSV is required (the table is not in the `docs/audio/celt/tables/`
//! set, which holds only tables the RFC does *not* inline). The
//! iCDF below is derived from the inlined PDF by the standard
//! `icdf[k] = (1<<ftb) − fh[k]` rule documented in §4.1.3.3.

use crate::range_decoder::RangeDecoder;

/// §4.3.3 number of cells in [`ALLOC_TRIM_PDF`] (RFC 6716 §4.3.3
/// p. 115, Table 58).
pub const ALLOC_TRIM_PDF_LEN: usize = 11;

/// §4.3.3 PDF denominator (RFC 6716 §4.3.3 p. 115, Table 58). The
/// table prescribes `{…}/128` so `ftb = log2(128) = 7`.
pub const ALLOC_TRIM_FTB: u32 = 7;

/// §4.3.3 PDF denominator value (`1 << ALLOC_TRIM_FTB = 128`).
pub const ALLOC_TRIM_PDF_DENOMINATOR: u32 = 1 << ALLOC_TRIM_FTB;

/// §4.3.3 default trim value (RFC 6716 §4.3.3 p. 114: "The default
/// value of 5 indicates no trim").
pub const ALLOC_TRIM_DEFAULT: u8 = 5;

/// §4.3.3 minimum trim value (RFC 6716 §4.3.3 p. 114: "an integer
/// value from 0-10").
pub const ALLOC_TRIM_MIN: u8 = 0;

/// §4.3.3 maximum trim value (RFC 6716 §4.3.3 p. 114: "an integer
/// value from 0-10").
pub const ALLOC_TRIM_MAX: u8 = 10;

/// §4.3.3 worst-case trim-symbol cost in 1/8 bits (RFC 6716 §4.3.3
/// p. 114: "plus 48 (6 bits)"). The trim symbol's range-coded budget
/// must fit in the remaining frame; the §4.3.3 gate compares
/// `ec_tell_frac() + ALLOC_TRIM_SIGNAL_COST_EIGHTH_BITS` against the
/// available 1/8-bit budget.
pub const ALLOC_TRIM_SIGNAL_COST_EIGHTH_BITS: u32 = 48;

/// §4.3.3 conversion factor: 1 byte = 8 whole bits = 64 1/8 bits.
pub const EIGHTH_BITS_PER_BYTE: u32 = 64;

/// §4.3.3 PDF for the trim (RFC 6716 §4.3.3 p. 115, Table 58:
/// `{2, 2, 5, 10, 22, 46, 22, 10, 5, 2, 2}/128`).
///
/// Symbol `k ∈ 0..=10` maps directly to the trim integer `k`. The
/// PDF is symmetric around `k = 5` (the default, `46/128`); cells
/// either side fall off as 22, 10, 5, 2, 2. The PDF sums to 128 =
/// [`ALLOC_TRIM_PDF_DENOMINATOR`].
pub const ALLOC_TRIM_PDF: [u8; ALLOC_TRIM_PDF_LEN] = [2, 2, 5, 10, 22, 46, 22, 10, 5, 2, 2];

/// §4.3.3 inverse CDF for the trim, derived from [`ALLOC_TRIM_PDF`]
/// by the §4.1.3.3 rule `icdf[k] = (1 << ftb) − fh[k]` with a
/// terminating zero.
///
/// Cumulative `fh = [2, 4, 9, 19, 41, 87, 109, 119, 124, 126, 128]`
/// (the running sum of [`ALLOC_TRIM_PDF`]), so
/// `icdf = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0]`. This is the
/// table consumed by [`RangeDecoder::dec_icdf`].
pub const ALLOC_TRIM_ICDF: [u8; ALLOC_TRIM_PDF_LEN] = [126, 124, 119, 109, 87, 41, 19, 9, 4, 2, 0];

/// Errors returned by [`decode_alloc_trim`] for inputs that violate
/// the §4.3.3 frame-budget bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocTrimError {
    /// `frame_size_bytes` overflows the 1/8-bit accounting space when
    /// scaled to 1/8 bits. The §4.3.3 budget is held in `u32`; the
    /// caller passed a `frame_size_bytes` that would exceed
    /// `u32::MAX / 64`. No Opus packet reaches this size.
    FrameSizeOverflows,
    /// `total_boost` already exceeds the §4.3.3 frame budget in 1/8
    /// bits. The §4.3.3 boost loop should have stopped well before
    /// the budget is consumed; a `total_boost` larger than the frame
    /// is a caller-side bug.
    TotalBoostExceedsFrame {
        frame_eighth_bits: u32,
        total_boost: u32,
    },
}

/// §4.3.3 signalling-gate predicate (RFC 6716 §4.3.3 p. 114).
///
/// Returns `true` iff
/// `ec_tell_frac + ALLOC_TRIM_SIGNAL_COST_EIGHTH_BITS ≤ frame_eighth_bits − total_boost`.
/// When the predicate is `false`, the trim is left at
/// [`ALLOC_TRIM_DEFAULT`] and no range-coder progress is made.
///
/// `ec_tell_frac` is the 1/8-bit-precision count of bits already
/// consumed by the range coder, returned by
/// [`RangeDecoder::tell_frac`]. `frame_eighth_bits` is the Opus frame
/// size in 1/8 bits (`bytes * 64`). `total_boost` is the 1/8-bit
/// boost accumulator from the §4.3.3 boost loop.
pub const fn alloc_trim_is_signalled(
    ec_tell_frac: u32,
    frame_eighth_bits: u32,
    total_boost: u32,
) -> bool {
    // Use saturating arithmetic so a malformed (boost > frame) input
    // produces the correct `false` result rather than wrapping.
    let budget = match frame_eighth_bits.checked_sub(total_boost) {
        Some(b) => b,
        None => return false,
    };
    match ec_tell_frac.checked_add(ALLOC_TRIM_SIGNAL_COST_EIGHTH_BITS) {
        Some(consumed_after) => consumed_after <= budget,
        None => false,
    }
}

/// Convert an Opus frame size in bytes to 1/8 bits, checking for
/// `u32` overflow.
///
/// The §4.3.3 budget bookkeeping is all in 1/8 bits per the RFC;
/// every accessor here takes its sizes in those units. A well-formed
/// Opus frame caps at 1275 bytes (§3.4 R5), so the result fits in
/// `u32` by a wide margin (1275 × 64 = 81600).
pub const fn frame_eighth_bits(frame_size_bytes: u32) -> Result<u32, AllocTrimError> {
    match frame_size_bytes.checked_mul(EIGHTH_BITS_PER_BYTE) {
        Some(v) => Ok(v),
        None => Err(AllocTrimError::FrameSizeOverflows),
    }
}

/// Decode the §4.3.3 allocation trim from the range coder (RFC 6716
/// §4.3.3 p. 114, Table 58 on p. 115).
///
/// Implements the §4.3.3 decode steps verbatim:
///
/// 1. Initialise `trim = ALLOC_TRIM_DEFAULT = 5`.
/// 2. Compute the §4.3.3 signalling gate
///    `[alloc_trim_is_signalled]` on the input
///    `(ec_tell_frac, frame_eighth_bits, total_boost)` triple.
/// 3. If the gate is satisfied, read one Table 58 symbol via
///    [`RangeDecoder::dec_icdf`] with iCDF [`ALLOC_TRIM_ICDF`] and
///    `ftb = ALLOC_TRIM_FTB`. The decoded symbol is the trim
///    integer, in `0..=10`.
/// 4. Return the trim as a [`u8`].
///
/// The `ec_tell_frac` argument is the 1/8-bit count of bits already
/// consumed by the range coder *before* this call (i.e.
/// `rd.tell_frac()` at the call site, after the §4.3.3 boost loop
/// finished). The §4.3.3 gate is evaluated against this value, then
/// the range coder is read iff the gate is satisfied; the gate is
/// not re-evaluated after the read.
///
/// Returns `Err` only for caller-side bookkeeping bugs
/// ([`AllocTrimError::FrameSizeOverflows`] /
/// [`AllocTrimError::TotalBoostExceedsFrame`]). A malformed range
/// stream is reported through the [`RangeDecoder`]'s sticky error
/// flag, not through this return type.
pub fn decode_alloc_trim(
    rd: &mut RangeDecoder<'_>,
    ec_tell_frac: u32,
    frame_size_bytes: u32,
    total_boost: u32,
) -> Result<u8, AllocTrimError> {
    let frame_eighth = frame_eighth_bits(frame_size_bytes)?;
    if total_boost > frame_eighth {
        return Err(AllocTrimError::TotalBoostExceedsFrame {
            frame_eighth_bits: frame_eighth,
            total_boost,
        });
    }
    if !alloc_trim_is_signalled(ec_tell_frac, frame_eighth, total_boost) {
        return Ok(ALLOC_TRIM_DEFAULT);
    }
    let symbol = rd.dec_icdf(&ALLOC_TRIM_ICDF, ALLOC_TRIM_FTB);
    // The §4.3.3 PDF has 11 cells; `dec_icdf` returns a value in
    // `0..ALLOC_TRIM_PDF_LEN`, which fits in `u8` and is by
    // construction within `0..=ALLOC_TRIM_MAX`.
    Ok(symbol as u8)
}

/// Borrow the full 11-byte [`ALLOC_TRIM_ICDF`] table.
///
/// Useful when a downstream sub-decoder wants to iterate the table
/// (or pin its full byte sequence in a regression test) without
/// re-indexing per call.
pub const fn alloc_trim_icdf() -> &'static [u8; ALLOC_TRIM_PDF_LEN] {
    &ALLOC_TRIM_ICDF
}

/// Borrow the full 11-byte [`ALLOC_TRIM_PDF`] table.
///
/// Useful for property tests that derive the iCDF from the PDF and
/// confirm the two tables stay in lockstep.
pub const fn alloc_trim_pdf() -> &'static [u8; ALLOC_TRIM_PDF_LEN] {
    &ALLOC_TRIM_PDF
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Table-shape constants ----

    #[test]
    fn pdf_length_constant_matches_array() {
        assert_eq!(ALLOC_TRIM_PDF_LEN, 11);
        assert_eq!(ALLOC_TRIM_PDF.len(), ALLOC_TRIM_PDF_LEN);
        assert_eq!(ALLOC_TRIM_ICDF.len(), ALLOC_TRIM_PDF_LEN);
    }

    #[test]
    fn ftb_constant_is_seven_for_denominator_128() {
        assert_eq!(ALLOC_TRIM_FTB, 7);
        assert_eq!(ALLOC_TRIM_PDF_DENOMINATOR, 128);
        assert_eq!(1u32 << ALLOC_TRIM_FTB, ALLOC_TRIM_PDF_DENOMINATOR);
    }

    #[test]
    fn trim_range_constants_match_rfc() {
        // RFC 6716 §4.3.3 p. 114: "an integer value from 0-10".
        assert_eq!(ALLOC_TRIM_MIN, 0);
        assert_eq!(ALLOC_TRIM_MAX, 10);
        // 11 valid integer values: 0,1,…,10.
        assert_eq!(
            (ALLOC_TRIM_MAX - ALLOC_TRIM_MIN + 1) as usize,
            ALLOC_TRIM_PDF_LEN
        );
    }

    #[test]
    fn default_trim_constant_matches_rfc() {
        // RFC 6716 §4.3.3 p. 114: "The default value of 5 indicates no
        // trim". Pinned because the gate-fail path returns this.
        assert_eq!(ALLOC_TRIM_DEFAULT, 5);
    }

    #[test]
    fn signal_cost_constant_matches_rfc() {
        // RFC 6716 §4.3.3 p. 114: "plus 48 (6 bits)" — 6 whole bits at
        // 1/8-bit precision is 48 units.
        assert_eq!(ALLOC_TRIM_SIGNAL_COST_EIGHTH_BITS, 48);
        assert_eq!(ALLOC_TRIM_SIGNAL_COST_EIGHTH_BITS / 8, 6);
    }

    #[test]
    fn eighth_bits_per_byte_constant_is_sixty_four() {
        // 1 byte = 8 whole bits = 64 1/8-bit units.
        assert_eq!(EIGHTH_BITS_PER_BYTE, 64);
    }

    // ---- Table 58 PDF cells pinned to the RFC body ----

    #[test]
    fn pdf_cells_match_table_58() {
        // RFC 6716 §4.3.3 p. 115, Table 58: `{2, 2, 5, 10, 22, 46, 22,
        // 10, 5, 2, 2}/128`.
        assert_eq!(ALLOC_TRIM_PDF, [2u8, 2, 5, 10, 22, 46, 22, 10, 5, 2, 2]);
    }

    #[test]
    fn pdf_sums_to_denominator() {
        let sum: u32 = ALLOC_TRIM_PDF.iter().map(|&c| c as u32).sum();
        assert_eq!(sum, ALLOC_TRIM_PDF_DENOMINATOR);
    }

    #[test]
    fn pdf_is_symmetric_around_default() {
        // Table 58 is symmetric around k=5 (the default). Pin the
        // §4.3.3 narrative's "lower bias / higher bias" symmetry.
        let n = ALLOC_TRIM_PDF.len();
        for k in 0..n {
            assert_eq!(
                ALLOC_TRIM_PDF[k],
                ALLOC_TRIM_PDF[n - 1 - k],
                "PDF not symmetric at k={}",
                k
            );
        }
    }

    #[test]
    fn default_cell_has_heaviest_mass() {
        // RFC 6716 §4.3.3 p. 114: "The default value of 5 indicates
        // no trim. The trim parameter is entropy coded in order to
        // lower the coding cost of less extreme adjustments." Pin
        // that the heaviest mass sits on the default.
        let max_idx = ALLOC_TRIM_PDF
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .map(|(k, _)| k)
            .expect("non-empty PDF");
        assert_eq!(max_idx, ALLOC_TRIM_DEFAULT as usize);
        assert_eq!(ALLOC_TRIM_PDF[max_idx], 46);
    }

    // ---- iCDF derivation cross-check ----

    #[test]
    fn icdf_matches_pdf_derivation() {
        // §4.1.3.3 rule: `icdf[k] = (1<<ftb) − fh[k]` with `fh` the
        // running cumulative sum of the PDF, terminating at `fh[K] =
        // 1<<ftb` (i.e. the last iCDF entry is 0).
        let mut fh: u32 = 0;
        for (k, &pmf) in ALLOC_TRIM_PDF.iter().enumerate() {
            fh += pmf as u32;
            let derived = ALLOC_TRIM_PDF_DENOMINATOR - fh;
            assert_eq!(
                ALLOC_TRIM_ICDF[k] as u32, derived,
                "iCDF/PDF mismatch at k={}",
                k
            );
        }
        assert_eq!(fh, ALLOC_TRIM_PDF_DENOMINATOR);
        assert_eq!(ALLOC_TRIM_ICDF[ALLOC_TRIM_PDF_LEN - 1], 0);
    }

    #[test]
    fn icdf_is_strictly_monotone_decreasing() {
        // The §4.1.3.3 contract: iCDF entries are strictly decreasing
        // up to the terminating zero.
        for w in ALLOC_TRIM_ICDF.windows(2) {
            assert!(w[0] > w[1], "iCDF not strictly decreasing at pair {:?}", w);
        }
    }

    #[test]
    fn icdf_spot_cells_pinned() {
        // Spot-pins so a future drift-edit on either PDF or iCDF
        // trips here even if the cross-check above is somehow
        // silenced.
        assert_eq!(ALLOC_TRIM_ICDF[0], 126); // 128 − 2
        assert_eq!(ALLOC_TRIM_ICDF[1], 124); // 128 − 4
        assert_eq!(ALLOC_TRIM_ICDF[5], 41); // 128 − 87 (default cell)
        assert_eq!(ALLOC_TRIM_ICDF[10], 0); // terminator
    }

    // ---- Borrow accessors mirror the const arrays ----

    #[test]
    fn pdf_borrow_returns_full_table() {
        assert_eq!(alloc_trim_pdf(), &ALLOC_TRIM_PDF);
    }

    #[test]
    fn icdf_borrow_returns_full_table() {
        assert_eq!(alloc_trim_icdf(), &ALLOC_TRIM_ICDF);
    }

    // ---- frame_eighth_bits ----

    #[test]
    fn frame_eighth_bits_scales_by_sixty_four() {
        assert_eq!(frame_eighth_bits(0).unwrap(), 0);
        assert_eq!(frame_eighth_bits(1).unwrap(), 64);
        // §3.4 R5: Opus frame max is 1275 bytes per channel mapping.
        assert_eq!(frame_eighth_bits(1275).unwrap(), 1275 * 64);
    }

    #[test]
    fn frame_eighth_bits_rejects_overflow() {
        // `u32::MAX / 64` is the boundary; anything above overflows.
        let boundary = u32::MAX / EIGHTH_BITS_PER_BYTE;
        // The boundary itself does *not* overflow (u32::MAX/64 * 64
        // fits exactly since 4 of u32::MAX's bits aren't multiples of
        // 64). One above the boundary does overflow.
        assert!(frame_eighth_bits(boundary).is_ok());
        assert_eq!(
            frame_eighth_bits(boundary + 1).unwrap_err(),
            AllocTrimError::FrameSizeOverflows
        );
        assert_eq!(
            frame_eighth_bits(u32::MAX).unwrap_err(),
            AllocTrimError::FrameSizeOverflows
        );
    }

    // ---- §4.3.3 signalling gate ----

    #[test]
    fn gate_passes_when_room_for_signal_cost() {
        // 60-byte Opus frame = 3840 1/8 bits. ec_tell_frac after the
        // boost loop hypothetically at 100; no boost. Gate should
        // pass: 100 + 48 = 148 ≤ 3840.
        assert!(alloc_trim_is_signalled(100, 3840, 0));
    }

    #[test]
    fn gate_fails_when_no_room() {
        // ec_tell_frac sits one 1/8 bit short of the budget; 48 more
        // overflows.
        let frame = 3840;
        assert!(!alloc_trim_is_signalled(frame - 47, frame, 0));
        // Equality is OK per the §4.3.3 "less than or equal to" rule.
        assert!(alloc_trim_is_signalled(frame - 48, frame, 0));
    }

    #[test]
    fn gate_subtracts_total_boost_from_budget() {
        // 60-byte frame = 3840 1/8 bits, but 100 1/8 bits already
        // committed to band boost. With ec_tell_frac=3700, gate sees
        // 3700+48 = 3748 vs 3840−100 = 3740 — fails.
        assert!(!alloc_trim_is_signalled(3700, 3840, 100));
        // ec_tell_frac=3691: 3691+48 = 3739 ≤ 3740 — passes.
        assert!(alloc_trim_is_signalled(3691, 3840, 100));
    }

    #[test]
    fn gate_handles_underflow_safely() {
        // total_boost greater than the frame should not panic.
        assert!(!alloc_trim_is_signalled(0, 100, 200));
    }

    #[test]
    fn gate_handles_ec_tell_frac_addition_overflow_safely() {
        // ec_tell_frac near u32::MAX must not wrap.
        assert!(!alloc_trim_is_signalled(u32::MAX, 3840, 0));
    }

    #[test]
    fn gate_at_six_bit_boundary() {
        // Exactly enough room for the six-bit symbol cost.
        // budget = frame_eighth_bits − total_boost. With frame=128
        // 1/8 bits and total_boost=0, budget=128. ec_tell_frac=80
        // gives 80+48=128 ≤ 128 — passes.
        assert!(alloc_trim_is_signalled(80, 128, 0));
        // One unit further is over the line.
        assert!(!alloc_trim_is_signalled(81, 128, 0));
    }

    // ---- decode_alloc_trim wrapper ----

    /// Build a range decoder over a stub payload. The exact bytes
    /// don't matter for the gate-fail path; the §4.3.3 gate is
    /// evaluated *before* the range coder is consulted.
    fn rd<'a>(buf: &'a [u8]) -> RangeDecoder<'a> {
        RangeDecoder::new(buf)
    }

    #[test]
    fn decode_returns_default_when_gate_fails() {
        // 1-byte frame, ec_tell_frac at 9 1/8 bits (after `new()`).
        // Frame = 64 1/8 bits; gate requires 9+48 = 57 ≤ 64−0 = 64 —
        // gate *passes*. To force gate failure: use a frame size of
        // 1 byte but a total_boost of 20.
        let buf = [0x80u8];
        let mut d = rd(&buf);
        // 64 − 20 = 44 budget; ec_tell_frac is 9; 9+48=57 > 44 → fail.
        let trim = decode_alloc_trim(&mut d, 9, 1, 20).unwrap();
        assert_eq!(trim, ALLOC_TRIM_DEFAULT);
    }

    #[test]
    fn decode_consumes_no_bits_when_gate_fails() {
        // The range coder state must be untouched on the gate-fail
        // path.
        let buf = [0x80u8];
        let mut d = rd(&buf);
        let before = d.tell();
        let _ = decode_alloc_trim(&mut d, 9, 1, 20).unwrap();
        assert_eq!(d.tell(), before);
    }

    #[test]
    fn decode_returns_in_range_value_when_gate_passes() {
        // Choose a payload + parameters where the gate definitely
        // passes; the returned trim must be in 0..=10 regardless of
        // the bit pattern.
        let buf = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut d = rd(&buf);
        let frac = d.tell_frac();
        let trim = decode_alloc_trim(&mut d, frac, 8, 0).unwrap();
        assert!(trim <= ALLOC_TRIM_MAX, "trim {} out of range", trim);
    }

    #[test]
    fn decode_consumes_range_bits_when_gate_passes() {
        let buf = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut d = rd(&buf);
        let before_frac = d.tell_frac();
        let _ = decode_alloc_trim(&mut d, before_frac, 8, 0).unwrap();
        // dec_icdf with ftb=7 + renormalization must advance the
        // 1/8-bit tell — we don't pin an exact number (it depends on
        // the symbol decoded for this exact bit pattern) but we do
        // pin that progress occurred.
        assert!(d.tell_frac() >= before_frac);
        // At least the 7-bit ftb budget should have been spent
        // (with renormalization on top).
        assert!(d.tell_frac() > before_frac);
    }

    #[test]
    fn decode_rejects_frame_size_overflow() {
        let buf = [0x80u8];
        let mut d = rd(&buf);
        let err = decode_alloc_trim(&mut d, 0, u32::MAX, 0).unwrap_err();
        assert_eq!(err, AllocTrimError::FrameSizeOverflows);
    }

    #[test]
    fn decode_rejects_boost_exceeding_frame() {
        let buf = [0x80u8];
        let mut d = rd(&buf);
        let err = decode_alloc_trim(&mut d, 0, 1, 100).unwrap_err();
        // 1 byte = 64 1/8 bits; total_boost 100 exceeds 64.
        assert_eq!(
            err,
            AllocTrimError::TotalBoostExceedsFrame {
                frame_eighth_bits: 64,
                total_boost: 100,
            }
        );
    }

    #[test]
    fn decode_does_not_consume_bits_on_error_paths() {
        let buf = [0x80u8];
        let mut d = rd(&buf);
        let before_frac = d.tell_frac();
        // Frame size overflow.
        let _ = decode_alloc_trim(&mut d, 0, u32::MAX, 0);
        assert_eq!(d.tell_frac(), before_frac);
        // Boost exceeds frame.
        let _ = decode_alloc_trim(&mut d, 0, 1, 100);
        assert_eq!(d.tell_frac(), before_frac);
    }

    // ---- Reachable-value sanity ----
    //
    // Every Table-58 cell is a reachable §4.3.3 decode outcome; the
    // outcome lands in `0..=10` and round-trips through the §4.3.3
    // narrative's integer-trim contract. Pin both ends.

    #[test]
    fn every_pdf_index_is_a_valid_trim_value() {
        // Every `k ∈ 0..ALLOC_TRIM_PDF_LEN` is in the §4.3.3 trim
        // integer range `ALLOC_TRIM_MIN..=ALLOC_TRIM_MAX`. Pinned as
        // an explicit `MIN..=MAX` window membership (avoids a
        // tautological lower-bound check now that `ALLOC_TRIM_MIN = 0
        // = u8::MIN`).
        for k in 0..ALLOC_TRIM_PDF_LEN as u8 {
            assert!((ALLOC_TRIM_MIN..=ALLOC_TRIM_MAX).contains(&k));
        }
    }

    #[test]
    fn pdf_index_count_equals_trim_value_count() {
        let trim_value_count = (ALLOC_TRIM_MAX - ALLOC_TRIM_MIN) as usize + 1;
        assert_eq!(trim_value_count, ALLOC_TRIM_PDF_LEN);
    }

    // ---- Gate vs. §4.3.3 worst case ----
    //
    // The §4.3.3 narrative's "plus 48 (6 bits)" is the worst-case cost
    // for one Table-58 symbol. The total mass on the worst-case cell
    // (`{2, 2, …, 2}/128`) gives an entropy of −log2(2/128) = 6 bits,
    // matching the gate's 48 1/8-bit budget. Pin the math.

    #[test]
    fn worst_case_symbol_cost_matches_gate_budget() {
        // Smallest cell mass is 2 / 128 → entropy 6 bits = 48 1/8.
        let min_mass: u32 = *ALLOC_TRIM_PDF.iter().min().unwrap() as u32;
        let denominator = ALLOC_TRIM_PDF_DENOMINATOR;
        assert_eq!(min_mass, 2);
        // log2(denominator / min_mass) = log2(64) = 6 whole bits.
        let bits = (denominator / min_mass).trailing_zeros();
        assert_eq!(bits, 6);
        assert_eq!(bits * 8, ALLOC_TRIM_SIGNAL_COST_EIGHTH_BITS);
    }
}
