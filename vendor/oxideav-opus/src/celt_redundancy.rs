//! CELT redundancy / mode-transition side-information decoder
//! (RFC 6716 §4.5.1, Tables 64 and 65).
//!
//! Mode transitions that change the audio bandwidth or switch the
//! lower-frequency model between SILK (LP) and CELT (MDCT) MAY
//! include a 5 ms redundant CELT frame as side information, so the
//! decoder does not have to invoke its packet-loss concealer across
//! the boundary. The bitstream signalling for that side-information
//! sits at the tail of every SILK-only or Hybrid Opus frame and uses
//! three sub-procedures:
//!
//! * §4.5.1.1 — the *redundancy flag* itself. SILK-only frames
//!   signal it implicitly (it's "on" iff the SILK portion left at
//!   least 17 unused bits in the Opus frame). Hybrid frames signal
//!   it explicitly with one Table 64 symbol (`{4095, 1}/4096`), but
//!   only when the SILK portion left at least 37 unused bits.
//! * §4.5.1.2 — the *redundancy position flag* (Table 65,
//!   `{1, 1}/2`). Decoded only when the redundancy flag is on. A
//!   value of zero places the redundant 5 ms CELT frame at the end
//!   of the Opus frame; a one places it at the beginning.
//! * §4.5.1.3 — the *redundancy size*. SILK-only: all remaining
//!   whole bytes belong to the redundant CELT frame (the §4.5.1.1
//!   "at least 17 bits remaining" rule guarantees ≥ 2 bytes). Hybrid:
//!   `2 + dec_uint(256)`. The decoded size may exceed the bytes that
//!   actually remain, in which case the Opus frame is invalid; the
//!   §4.5.1.3 RECOMMENDATION is to stop decoding and discard the
//!   rest of the Opus frame.
//!
//! CELT-only Opus frames never carry redundancy side information per
//! §4.5.1: the redundant frame is only relevant when the *next*
//! frame after a transition is SILK-only or Hybrid (or the previous
//! frame before a transition is SILK-only or Hybrid). The decoder
//! that owns this module is expected to skip the redundancy step
//! entirely for [`OperatingMode::CeltOnly`].
//!
//! No external library source was consulted; every PDF, every
//! threshold (17 bits, 37 bits, +2 bytes, 256-bucket `dec_uint`),
//! and every conditional is transcribed from RFC 6716 §4.5.1
//! (`docs/audio/opus/rfc6716-opus.txt`, pp. 124–126).

use crate::framing::OperatingMode;
use crate::range_decoder::RangeDecoder;

/// The §4.5.1.1 minimum-bits gate for an *implicit* SILK-only
/// redundancy signal.
///
/// SILK-only Opus frames carry the redundancy flag implicitly: after
/// the SILK decoder finishes, the decoder checks whether at least
/// `17` bits of the Opus frame remained unconsumed. If so, the frame
/// is treated as if a redundancy flag of "1" had been signalled and
/// the trailing bytes ARE the redundant CELT frame; if not, no
/// redundancy is present.
pub const SILK_ONLY_REDUNDANCY_MIN_REMAINING_BITS: u32 = 17;

/// The §4.5.1.1 minimum-bits gate for an *explicit* Hybrid
/// redundancy signal.
///
/// Hybrid Opus frames carry the redundancy flag explicitly via the
/// Table 64 symbol. To leave room for both the redundancy bit itself
/// AND a minimum-size 2-byte redundant CELT frame, the §4.5.1.1
/// procedure only decodes the Table 64 symbol if at least `37` bits
/// remained in the Opus frame after the SILK decode.
pub const HYBRID_REDUNDANCY_MIN_REMAINING_BITS: u32 = 37;

/// Inverse-CDF form of the §4.5.1.1 Table 64 redundancy flag PDF
/// (Hybrid frames only). PDF is `{4095, 1}/4096` (ftb = 12), so
/// `icdf = {4096 - 4095, 4096 - 4096} = {1, 0}` and the symbol space
/// is the two-cell partition `{0, 1}` with `1` rare.
pub const REDUNDANCY_FLAG_ICDF_FTB: u32 = 12;
/// Companion inverse-CDF table for [`REDUNDANCY_FLAG_ICDF_FTB`].
pub const REDUNDANCY_FLAG_ICDF: [u8; 2] = [1, 0];

/// Inverse-CDF form of the §4.5.1.2 Table 65 redundancy position
/// PDF. PDF is `{1, 1}/2` (the uniform binary distribution), so
/// `icdf = {2 - 1, 2 - 2} = {1, 0}` with `ftb = 1`. Decoded only when
/// the redundancy flag is set.
pub const REDUNDANCY_POSITION_ICDF_FTB: u32 = 1;
/// Companion inverse-CDF table for [`REDUNDANCY_POSITION_ICDF_FTB`].
pub const REDUNDANCY_POSITION_ICDF: [u8; 2] = [1, 0];

/// The §4.5.1.3 Hybrid baseline offset added to the `dec_uint(256)`
/// payload size.
pub const HYBRID_REDUNDANCY_SIZE_BASELINE_BYTES: usize = 2;

/// The §4.5.1.3 Hybrid `dec_uint` payload alphabet size.
pub const HYBRID_REDUNDANCY_SIZE_DEC_UINT_FT: u32 = 256;

/// The §4.5.1.3 minimum size of a redundant CELT frame.
///
/// Both modes pin the same lower bound — Hybrid because the §4.5.1.3
/// arithmetic adds a baseline of 2 to a non-negative `dec_uint`
/// value, SILK-only because the §4.5.1.1 17-bit remaining-budget
/// gate is sized so that ≥ 2 whole bytes are left when the implicit
/// flag fires.
pub const REDUNDANCY_MIN_SIZE_BYTES: usize = 2;

/// Where the 5 ms redundant CELT frame is placed inside the Opus
/// frame, per RFC 6716 §4.5.1.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedundancyPosition {
    /// Redundant frame is at the *end* of the Opus frame. Table 65
    /// symbol = 0. Per §4.5.1.2 this signals "first frame in the
    /// transition" (the redundant CELT frame belongs to the trailing
    /// edge, where the next Opus frame's CELT layer begins).
    End,
    /// Redundant frame is at the *beginning* of the Opus frame.
    /// Table 65 symbol = 1. Per §4.5.1.2 this signals "second frame
    /// in the transition" (the redundant CELT frame belongs to the
    /// leading edge, where the previous Opus frame's CELT layer
    /// ended).
    Beginning,
}

impl RedundancyPosition {
    /// Decode the Table 65 symbol into a [`RedundancyPosition`].
    pub fn from_symbol(symbol: u32) -> Self {
        if symbol == 0 {
            RedundancyPosition::End
        } else {
            RedundancyPosition::Beginning
        }
    }
}

/// Outcome of the §4.5.1 redundancy decode for one Opus frame.
///
/// Variants encode the three legal outcomes:
///
/// * [`RedundancyDecision::NotPresent`] — no redundant CELT frame is
///   embedded in this Opus frame. CELT-only Opus frames always land
///   here (the decoder bypasses the entire §4.5.1 path); SILK-only
///   and Hybrid Opus frames land here when the §4.5.1.1 remaining-
///   bits gate failed or the Hybrid Table 64 symbol read as zero.
/// * [`RedundancyDecision::Present`] — the redundant CELT frame is
///   present. Carries the §4.5.1.2 position and the §4.5.1.3 size in
///   whole bytes.
/// * [`RedundancyDecision::Invalid`] — the §4.5.1.3 size computation
///   produced a value larger than the bytes actually available in
///   the Opus frame buffer. Per §4.5.1.3, "a decoder is not required
///   to ignore the entire frame ... it is RECOMMENDED that the
///   decoder stop decoding and discard the rest of the current Opus
///   frame." The caller can drop any partial CELT output and proceed
///   with whatever audio was decoded prior to the §4.5.1 stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedundancyDecision {
    /// No redundant CELT frame in this Opus frame.
    #[default]
    NotPresent,
    /// Redundant CELT frame is present at `position`, sized
    /// `size_bytes` whole bytes (always at least
    /// [`REDUNDANCY_MIN_SIZE_BYTES`]).
    Present {
        position: RedundancyPosition,
        size_bytes: usize,
    },
    /// §4.5.1.3 size exceeds the bytes remaining in the Opus frame
    /// buffer.
    Invalid,
}

impl RedundancyDecision {
    /// `true` iff the decision is [`Self::Present`].
    pub fn is_present(&self) -> bool {
        matches!(self, RedundancyDecision::Present { .. })
    }

    /// Position of the redundant CELT frame, if any.
    pub fn position(&self) -> Option<RedundancyPosition> {
        match self {
            RedundancyDecision::Present { position, .. } => Some(*position),
            _ => None,
        }
    }

    /// Size (in whole bytes) of the redundant CELT frame, if any.
    pub fn size_bytes(&self) -> Option<usize> {
        match self {
            RedundancyDecision::Present { size_bytes, .. } => Some(*size_bytes),
            _ => None,
        }
    }
}

/// Compute the §4.5.1.1 remaining-bits budget for an Opus frame
/// after the SILK portion has been decoded.
///
/// Returns the number of bits in the Opus frame buffer not yet
/// consumed by the range coder or by raw bits. The argument is the
/// total size of the Opus frame *payload* in bytes — what
/// [`RangeDecoder::new`] was originally constructed with — which the
/// caller knows from the §3.2 frame-packing parse (a slice length
/// out of [`crate::OpusPacket`]).
///
/// Multiplying by 8 is the §4.1.6 convention; raw bits and
/// range-coded bits both count against the same budget via
/// [`RangeDecoder::tell`].
pub fn remaining_bits(rd: &RangeDecoder<'_>, opus_frame_bytes: usize) -> u32 {
    let total = (opus_frame_bytes as u64).saturating_mul(8);
    let tell = rd.tell() as u64;
    total.saturating_sub(tell).min(u32::MAX as u64) as u32
}

/// Number of whole bytes remaining in the Opus frame after the SILK
/// (and, for Hybrid frames, the redundancy-flag and -position)
/// decoding steps. Per §4.5.1.3, "the number of bytes in the
/// redundant CELT frame is simply the number of whole bytes
/// remaining" — for SILK-only frames this is the redundancy size
/// directly; for Hybrid frames the §4.5.1.3 `2 + dec_uint(256)`
/// value is checked against this number for the validity test.
pub fn whole_bytes_remaining(rd: &RangeDecoder<'_>, opus_frame_bytes: usize) -> usize {
    let remaining = remaining_bits(rd, opus_frame_bytes);
    (remaining / 8) as usize
}

/// Decode the §4.5.1 redundancy side information at the tail of an
/// Opus frame.
///
/// Driver layout per RFC 6716 §4.5.1:
///
/// 1. CELT-only Opus frames have no §4.5.1 side information; caller
///    should not invoke this function (we still handle the call
///    correctly and return [`RedundancyDecision::NotPresent`]).
/// 2. SILK-only Opus frames: if `remaining_bits >= 17`, redundancy is
///    implicitly on, position is decoded from Table 65, and the
///    redundant size is exactly the remaining whole bytes.
/// 3. Hybrid Opus frames: if `remaining_bits >= 37`, decode the
///    Table 64 flag; if it reads 1, decode the Table 65 position and
///    compute the redundancy size as `2 + dec_uint(256)`. If that
///    exceeds the whole bytes available, the decision is
///    [`RedundancyDecision::Invalid`].
///
/// `opus_frame_bytes` is the size in bytes of the *original Opus
/// frame buffer* the range decoder was constructed from, used to
/// compute "remaining bits" and "remaining whole bytes" per
/// §4.1.6 / §4.5.1.3.
pub fn decode_redundancy(
    rd: &mut RangeDecoder<'_>,
    mode: OperatingMode,
    opus_frame_bytes: usize,
) -> RedundancyDecision {
    match mode {
        // §4.5.1: redundancy is signalled only in SILK-only and Hybrid
        // Opus frames. CELT-only frames never carry it.
        OperatingMode::CeltOnly => RedundancyDecision::NotPresent,
        OperatingMode::SilkOnly => decode_silk_only(rd, opus_frame_bytes),
        OperatingMode::Hybrid => decode_hybrid(rd, opus_frame_bytes),
    }
}

fn decode_silk_only(rd: &mut RangeDecoder<'_>, opus_frame_bytes: usize) -> RedundancyDecision {
    // §4.5.1.1: implicit signalling. The redundancy flag is "on" iff
    // at least 17 bits remain after the SILK decoder finished.
    if remaining_bits(rd, opus_frame_bytes) < SILK_ONLY_REDUNDANCY_MIN_REMAINING_BITS {
        return RedundancyDecision::NotPresent;
    }
    // §4.5.1.2: position is the Table 65 1-bit uniform symbol.
    let pos_symbol = rd.dec_icdf(&REDUNDANCY_POSITION_ICDF, REDUNDANCY_POSITION_ICDF_FTB);
    let position = RedundancyPosition::from_symbol(pos_symbol);
    // §4.5.1.3: redundant size is the remaining whole bytes after the
    // §4.5.1.2 position read. The 17-bit guard ensures this is at
    // least 2 (16 bits of the 17-bit window, after consuming up to a
    // single 1-bit position read, still leaves a full byte plus some
    // — and the redundant frame is byte-aligned per §4.5.1.3).
    let size_bytes = whole_bytes_remaining(rd, opus_frame_bytes);
    if size_bytes < REDUNDANCY_MIN_SIZE_BYTES {
        // Defensive: the §4.5.1.1 17-bit gate is sized so this branch
        // is unreachable for a well-formed buffer. If a corrupt frame
        // somehow falls through, report Invalid rather than synthesise
        // a too-short redundant CELT slice.
        return RedundancyDecision::Invalid;
    }
    RedundancyDecision::Present {
        position,
        size_bytes,
    }
}

fn decode_hybrid(rd: &mut RangeDecoder<'_>, opus_frame_bytes: usize) -> RedundancyDecision {
    // §4.5.1.1: Hybrid signalling is explicit but gated by a 37-bit
    // remaining budget so the redundancy flag itself plus a minimum
    // 2-byte redundant frame still fit.
    if remaining_bits(rd, opus_frame_bytes) < HYBRID_REDUNDANCY_MIN_REMAINING_BITS {
        return RedundancyDecision::NotPresent;
    }
    // §4.5.1.1 Table 64 flag.
    let flag = rd.dec_icdf(&REDUNDANCY_FLAG_ICDF, REDUNDANCY_FLAG_ICDF_FTB);
    if flag == 0 {
        return RedundancyDecision::NotPresent;
    }
    // §4.5.1.2 Table 65 position.
    let pos_symbol = rd.dec_icdf(&REDUNDANCY_POSITION_ICDF, REDUNDANCY_POSITION_ICDF_FTB);
    let position = RedundancyPosition::from_symbol(pos_symbol);
    // §4.5.1.3 size: 2 + dec_uint(256). The dec_uint() consumes range-
    // coded + raw bits per §4.1.5.
    let payload = match rd.dec_uint(HYBRID_REDUNDANCY_SIZE_DEC_UINT_FT) {
        Ok(v) => v as usize,
        Err(_) => return RedundancyDecision::Invalid,
    };
    let claimed_size = HYBRID_REDUNDANCY_SIZE_BASELINE_BYTES.saturating_add(payload);
    let whole_bytes = whole_bytes_remaining(rd, opus_frame_bytes);
    if claimed_size > whole_bytes {
        // §4.5.1.3: "This may be more than the number of whole bytes
        // remaining in the Opus frame, in which case the frame is
        // invalid."
        return RedundancyDecision::Invalid;
    }
    RedundancyDecision::Present {
        position,
        size_bytes: claimed_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: construct a RangeDecoder from a byte slice. The
    /// returned decoder borrows the slice for its lifetime.
    fn rd<'a>(buf: &'a [u8]) -> RangeDecoder<'a> {
        RangeDecoder::new(buf)
    }

    #[test]
    fn celt_only_always_not_present() {
        // §4.5.1 explicitly excludes CELT-only frames from
        // redundancy signalling; the decision is constant regardless
        // of the buffer.
        let buf = [0x80, 0x00, 0x00, 0x00];
        let mut decoder = rd(&buf);
        let decision = decode_redundancy(&mut decoder, OperatingMode::CeltOnly, buf.len());
        assert_eq!(decision, RedundancyDecision::NotPresent);
        // Sanity: nothing was consumed from the range coder.
        assert_eq!(decoder.tell(), 1);
    }

    #[test]
    fn silk_only_below_17_bits_remaining_is_not_present() {
        // A 2-byte buffer has 16 total bits. After init `tell() == 1`,
        // leaving 15 bits remaining — below the 17-bit gate. The
        // implicit flag is OFF; no redundancy.
        let buf = [0xAA, 0x55];
        let mut decoder = rd(&buf);
        // Force tell() to be larger than 0 (the constructor already
        // initialises it to 1 per §4.1.6).
        let decision = decode_redundancy(&mut decoder, OperatingMode::SilkOnly, buf.len());
        assert_eq!(decision, RedundancyDecision::NotPresent);
    }

    #[test]
    fn silk_only_with_full_buffer_is_present() {
        // A fresh 8-byte buffer has 64 bits total; after init `tell()`
        // is 1, so 63 bits remain — well above the 17-bit gate. The
        // implicit flag is ON.
        let buf = [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let mut decoder = rd(&buf);
        let decision = decode_redundancy(&mut decoder, OperatingMode::SilkOnly, buf.len());
        assert!(decision.is_present());
        let size = decision.size_bytes().unwrap();
        // At least 2 bytes per §4.5.1.3 invariant.
        assert!(size >= REDUNDANCY_MIN_SIZE_BYTES);
        // No bigger than the whole buffer.
        assert!(size <= buf.len());
    }

    #[test]
    fn silk_only_size_eq_whole_bytes_remaining_invariant() {
        // For SILK-only frames §4.5.1.3 reports the redundancy size
        // is exactly the whole bytes remaining after the §4.5.1.2
        // position bit was decoded.
        let buf = [0x55, 0xAA, 0xC3, 0x3C, 0x96, 0x69];
        let mut decoder = rd(&buf);
        let pre_decision_decoder_clone_tell;
        // First, snapshot what the decoder's tell will look like
        // *after* the §4.5.1.2 position read (one Table 65 ICDF):
        {
            let mut probe = rd(&buf);
            // Skip nothing; decoder is fresh, just like the real call.
            let _ = probe.dec_icdf(&REDUNDANCY_POSITION_ICDF, REDUNDANCY_POSITION_ICDF_FTB);
            pre_decision_decoder_clone_tell = whole_bytes_remaining(&probe, buf.len());
        }
        let decision = decode_redundancy(&mut decoder, OperatingMode::SilkOnly, buf.len());
        if let RedundancyDecision::Present { size_bytes, .. } = decision {
            assert_eq!(size_bytes, pre_decision_decoder_clone_tell);
        } else {
            panic!("expected SILK-only redundancy present, got {:?}", decision);
        }
    }

    #[test]
    fn hybrid_below_37_bits_remaining_is_not_present() {
        // 4 bytes = 32 bits; well below the 37-bit gate. No flag is
        // decoded.
        let buf = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut decoder = rd(&buf);
        let decision = decode_redundancy(&mut decoder, OperatingMode::Hybrid, buf.len());
        assert_eq!(decision, RedundancyDecision::NotPresent);
    }

    #[test]
    fn hybrid_with_full_buffer_decodes_flag() {
        // 16 bytes = 128 bits; above the 37-bit gate. The Table 64
        // flag is decoded and the decision is one of NotPresent /
        // Present / Invalid depending on the buffer contents.
        let buf = [0u8; 16];
        let mut decoder = rd(&buf);
        let decision = decode_redundancy(&mut decoder, OperatingMode::Hybrid, buf.len());
        // The Table 64 PDF is {4095, 1}/4096 with the "1" case rare.
        // An all-zero buffer takes the more-common 0-symbol path.
        // We do not pin the outcome to a value but verify the decoder
        // ran without panicking and `tell()` advanced past the flag.
        assert!(decoder.tell() > 1);
        let _ = decision;
    }

    #[test]
    fn icdf_tables_match_rfc_pdfs() {
        // §4.5.1.1 Table 64: PDF = {4095, 1}/4096 with ftb = 12.
        // ICDF[k] = ft - fh[k] = {4096-4095, 4096-4096} = {1, 0}.
        assert_eq!(REDUNDANCY_FLAG_ICDF, [1u8, 0]);
        assert_eq!(REDUNDANCY_FLAG_ICDF_FTB, 12);
        // §4.5.1.2 Table 65: PDF = {1, 1}/2 with ftb = 1.
        // ICDF = {2-1, 2-2} = {1, 0}.
        assert_eq!(REDUNDANCY_POSITION_ICDF, [1u8, 0]);
        assert_eq!(REDUNDANCY_POSITION_ICDF_FTB, 1);
    }

    #[test]
    fn position_symbol_round_trip() {
        // §4.5.1.2 spec text: "If the value is zero, this is the
        // first frame in the transition, and the redundancy belongs
        // at the end. If the value is one, this is the second frame
        // in the transition, and the redundancy belongs at the
        // beginning."
        assert_eq!(RedundancyPosition::from_symbol(0), RedundancyPosition::End);
        assert_eq!(
            RedundancyPosition::from_symbol(1),
            RedundancyPosition::Beginning
        );
        // Defensive: out-of-range values (which the range decoder
        // cannot actually produce for a 2-symbol PDF) fall through to
        // Beginning. This matches the documented "1 = beginning"
        // pole, since any non-zero symbol implies non-zero
        // signalling.
        assert_eq!(
            RedundancyPosition::from_symbol(42),
            RedundancyPosition::Beginning
        );
    }

    #[test]
    fn remaining_bits_matches_total_minus_tell() {
        let buf = [0xFFu8; 5];
        let decoder = rd(&buf);
        let total_bits = (buf.len() as u32) * 8;
        let tell = decoder.tell();
        assert_eq!(remaining_bits(&decoder, buf.len()), total_bits - tell);
    }

    #[test]
    fn whole_bytes_remaining_floors_to_bytes() {
        // 5 bytes = 40 total bits; after init tell() = 1 leaves 39
        // remaining bits = floor(39/8) = 4 whole bytes.
        let buf = [0u8; 5];
        let decoder = rd(&buf);
        assert_eq!(whole_bytes_remaining(&decoder, buf.len()), 4);
    }

    #[test]
    fn decision_helpers_consistent() {
        let absent = RedundancyDecision::NotPresent;
        assert!(!absent.is_present());
        assert_eq!(absent.position(), None);
        assert_eq!(absent.size_bytes(), None);
        let present = RedundancyDecision::Present {
            position: RedundancyPosition::Beginning,
            size_bytes: 7,
        };
        assert!(present.is_present());
        assert_eq!(present.position(), Some(RedundancyPosition::Beginning));
        assert_eq!(present.size_bytes(), Some(7));
        let invalid = RedundancyDecision::Invalid;
        assert!(!invalid.is_present());
        assert_eq!(invalid.position(), None);
        assert_eq!(invalid.size_bytes(), None);
    }

    #[test]
    fn constants_match_rfc_text() {
        // §4.5.1.1 thresholds.
        assert_eq!(SILK_ONLY_REDUNDANCY_MIN_REMAINING_BITS, 17);
        assert_eq!(HYBRID_REDUNDANCY_MIN_REMAINING_BITS, 37);
        // §4.5.1.3 Hybrid size formula constants.
        assert_eq!(HYBRID_REDUNDANCY_SIZE_BASELINE_BYTES, 2);
        assert_eq!(HYBRID_REDUNDANCY_SIZE_DEC_UINT_FT, 256);
        // §4.5.1.3 minimum size.
        assert_eq!(REDUNDANCY_MIN_SIZE_BYTES, 2);
    }
}
