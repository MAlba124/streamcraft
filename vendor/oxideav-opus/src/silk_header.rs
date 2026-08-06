//! SILK packet-level header bits — RFC 6716 §4.2.3 and §4.2.4.
//!
//! Sits *between* the §3 packet framing and the per-SILK-frame body
//! decoded by [`crate::silk_frame`]. This module owns the two
//! packet-level decisions that govern how many SILK frames the
//! downstream decoder must decode (and which of those are LBRR
//! redundancy copies):
//!
//! * **§4.2.3 Header Bits** — for each channel (mono: 1; stereo: 2),
//!   N Voice-Activity-Detection (VAD) bits followed by one global LBRR
//!   flag, where N is the number of SILK frames in the Opus frame
//!   (1 for a 10/20 ms Opus frame, 2 for 40 ms, 3 for 60 ms; CELT-only
//!   2.5 / 5 ms Opus frames have no SILK header bits and are out of
//!   scope here). All these flags are coded as uniform binary symbols
//!   with `{1, 1}/2` per Table 3, so each one is a single
//!   `dec_bit_logp(1)` call.
//! * **§4.2.4 Per-Frame LBRR Flags** — only present on Opus frames
//!   strictly longer than 20 ms (i.e. 40 ms and 60 ms), and only for
//!   channels whose global LBRR flag from §4.2.3 is set. Decoded as a
//!   single symbol from Table 4 (PDF `{0, 53, 53, 150}/256` for 40 ms
//!   and `{0, 41, 20, 29, 41, 15, 28, 82}/256` for 60 ms). The decoded
//!   value is a 2- or 3-bit integer carrying the per-SILK-frame LBRR
//!   flags packed from the LSB to the MSB; bit 0 corresponds to SILK
//!   frame 0.
//!
//! The output ([`SilkHeaderBits`]) records both the per-channel VAD
//! pattern and a fully expanded per-channel × per-SILK-frame LBRR
//! bitmap. The order of LBRR frame appearance in the bitstream
//! (mono / stereo, longer / shorter than 20 ms) follows §4.2.2
//! Figures 15 and 16, but that layout is the caller's concern — this
//! module only decodes the header bits themselves.
//!
//! ## Scope
//!
//! Per §4.2.3 the VAD + LBRR header bits are "the first symbols
//! decoded by the range coder" for any LP-bearing Opus frame, so this
//! module is the entry point downstream of [`crate::range_decoder`]
//! and [`crate::frames`]. The §4.2.5 LBRR frame body and §4.2.6
//! regular SILK frame body decoding stays in [`crate::silk_frame`]
//! and friends.
//!
//! All PDF transcriptions and the bit-packing convention are taken
//! verbatim from RFC 6716. No external library source consulted.

use crate::range_decoder::RangeDecoder;
use crate::Error;

/// Maximum number of SILK frames in a single channel of one Opus
/// frame, per §4.2.2 (three 20 ms SILK frames in a 60 ms Opus frame).
pub const SILK_MAX_FRAMES_PER_CHANNEL: usize = 3;

/// Compute the number of 20 ms (or single 10 ms) SILK frames a given
/// Opus frame size contains, per §4.2.2 "LP Layer Organization".
///
/// Maps the §3.1 Table 2 `frame_size_tenths_ms` value to the SILK
/// frame count:
///
/// * 100 (10 ms) ⇒ 1
/// * 200 (20 ms) ⇒ 1
/// * 400 (40 ms) ⇒ 2
/// * 600 (60 ms) ⇒ 3
/// * any 2.5 / 5 ms frame (CELT-only, no SILK layer) ⇒ `None`.
///
/// Callers that have already screened for a SILK-bearing mode can
/// `.unwrap()` the result; the `None` arm exists so a caller that
/// hands an arbitrary TOC frame size in doesn't have to do its own
/// filtering.
pub fn silk_frame_count(frame_size_tenths_ms: u16) -> Option<u8> {
    match frame_size_tenths_ms {
        100 | 200 => Some(1),
        400 => Some(2),
        600 => Some(3),
        // 25 / 50 = 2.5 / 5 ms are CELT-only and have no SILK header.
        _ => None,
    }
}

/// Header bits for one channel (mid or side), §4.2.3.
///
/// Holds the VAD flags as a packed bitmap (`bit i` = VAD flag for
/// SILK frame `i`) and the single global LBRR flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilkChannelHeader {
    /// VAD bitmap. Only the low `num_silk_frames` bits are meaningful;
    /// higher bits are always 0. Bit 0 = VAD flag for SILK frame 0
    /// (the earliest in time). A set bit means the frame was coded as
    /// active speech.
    pub vad_flags: u8,
    /// §4.2.3 global LBRR flag for this channel. When set, an
    /// per-channel LBRR set is present in §4.2.4 for Opus frames
    /// longer than 20 ms, and at least one LBRR frame body follows
    /// (per the LBRR bitmap, or single LBRR frame for ≤ 20 ms).
    pub lbrr_flag: bool,
}

/// Per-SILK-frame LBRR bitmap for one channel, §4.2.4.
///
/// `bit i` = LBRR flag for SILK frame `i`. Only the low
/// `num_silk_frames` bits are meaningful; higher bits are 0. For a
/// channel whose §4.2.3 global LBRR flag is unset, this bitmap is
/// zero; for a 10 / 20 ms Opus frame with the global LBRR flag set,
/// bit 0 is set (a single LBRR frame is implied by §4.2.3 directly,
/// per the RFC's "global LBRR flag in the header bits is already
/// sufficient" wording).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PerFrameLbrr {
    /// Mid channel (also used as the sole channel in mono).
    pub mid: u8,
    /// Side channel (always 0 for mono).
    pub side: u8,
}

/// Decoded §4.2.3 + §4.2.4 packet-level header bits for one Opus
/// frame that carries a SILK layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilkHeaderBits {
    /// Number of SILK frames per channel (1, 2, or 3) for this Opus
    /// frame, per §4.2.2.
    pub num_silk_frames: u8,
    /// Mid-channel header. In mono Opus frames this is the only
    /// channel; in stereo it is the mid channel of the mid/side pair.
    pub mid: SilkChannelHeader,
    /// Side-channel header. `None` for mono Opus frames; `Some(_)`
    /// for stereo Opus frames.
    pub side: Option<SilkChannelHeader>,
    /// Per-SILK-frame LBRR bitmap for both channels, derived from
    /// §4.2.4 (Opus frames > 20 ms) or implied by §4.2.3 (Opus
    /// frames ≤ 20 ms).
    pub per_frame_lbrr: PerFrameLbrr,
}

impl SilkHeaderBits {
    /// Decode the §4.2.3 header bits + §4.2.4 per-frame LBRR flags
    /// from `rd` for an Opus frame that carries SILK.
    ///
    /// `num_silk_frames` must be 1, 2, or 3 (see [`silk_frame_count`]);
    /// any other value yields [`Error::MalformedPacket`]. `stereo`
    /// selects the §4.2.3 single-channel (mono) vs two-channel
    /// (stereo) header layout.
    ///
    /// Bitstream order per §4.2.2 Figures 15/16:
    ///
    /// 1. Mid VAD flags (N bits) + Mid LBRR flag (1 bit).
    /// 2. (Stereo only) Side VAD flags (N bits) + Side LBRR flag.
    /// 3. (Opus frame > 20 ms only) Mid per-frame LBRR set, then
    ///    Side per-frame LBRR set (if stereo and Side LBRR is set),
    ///    each decoded via [`per_frame_lbrr_pdf`].
    ///
    /// All header-bit symbols are uniform binary
    /// (`dec_bit_logp(1)`); the per-frame LBRR sets are §4.1.3.3
    /// inverse-CDF reads against the Table 4 PDFs.
    pub fn decode(
        rd: &mut RangeDecoder<'_>,
        num_silk_frames: u8,
        stereo: bool,
    ) -> Result<Self, Error> {
        if !(1..=3).contains(&num_silk_frames) {
            return Err(Error::MalformedPacket);
        }

        // §4.2.3 step 1 — mid channel VAD bits + LBRR flag.
        let mid = decode_channel_header(rd, num_silk_frames);

        // §4.2.3 step 2 — side channel (stereo only).
        let side = if stereo {
            Some(decode_channel_header(rd, num_silk_frames))
        } else {
            None
        };

        // §4.2.4 — per-frame LBRR flag set, when the Opus frame is
        // strictly longer than 20 ms (num_silk_frames >= 2) AND the
        // corresponding channel's global LBRR flag is set.
        //
        // For an Opus frame of 10 or 20 ms (num_silk_frames == 1),
        // the global LBRR flag itself acts as the per-frame LBRR flag
        // for SILK frame 0 ("the global LBRR flag in the header bits
        // is already sufficient to indicate the presence of that
        // single LBRR frame").
        let per_frame_lbrr = if num_silk_frames == 1 {
            PerFrameLbrr {
                mid: u8::from(mid.lbrr_flag),
                side: side.as_ref().map(|s| u8::from(s.lbrr_flag)).unwrap_or(0),
            }
        } else {
            let mid_bits = if mid.lbrr_flag {
                decode_per_frame_lbrr(rd, num_silk_frames)
            } else {
                0
            };
            let side_bits = match side {
                Some(s) if s.lbrr_flag => decode_per_frame_lbrr(rd, num_silk_frames),
                _ => 0,
            };
            PerFrameLbrr {
                mid: mid_bits,
                side: side_bits,
            }
        };

        Ok(Self {
            num_silk_frames,
            mid,
            side,
            per_frame_lbrr,
        })
    }

    /// Encode this header-bit set into `re` — the exact write-side
    /// mirror of [`Self::decode`] (§4.2.3 / §4.2.4 bitstream order).
    ///
    /// Validates internal consistency: `num_silk_frames ∈ 1..=3`,
    /// VAD bitmaps confined to the low `num_silk_frames` bits, and —
    /// when `num_silk_frames >= 2` — a per-frame LBRR bitmap that is
    /// non-zero iff the channel's global LBRR flag is set (Table 4
    /// excludes the all-zero value). For `num_silk_frames == 1` the
    /// per-frame bitmap must equal the global flag (no symbol is
    /// written; §4.2.4).
    pub fn encode(&self, re: &mut crate::range_encoder::RangeEncoder) -> Result<(), Error> {
        let n = self.num_silk_frames;
        if !(1..=3).contains(&n) {
            return Err(Error::MalformedPacket);
        }
        let mask = (1u8 << n) - 1;

        let encode_channel = |re: &mut crate::range_encoder::RangeEncoder,
                              ch: &SilkChannelHeader|
         -> Result<(), Error> {
            if ch.vad_flags & !mask != 0 {
                return Err(Error::MalformedPacket);
            }
            for i in 0..n {
                re.enc_bit_logp((ch.vad_flags >> i) & 1 == 1, 1);
            }
            re.enc_bit_logp(ch.lbrr_flag, 1);
            Ok(())
        };

        // §4.2.3 step 1 — mid channel VAD bits + LBRR flag.
        encode_channel(re, &self.mid)?;
        // §4.2.3 step 2 — side channel (stereo only).
        if let Some(side) = &self.side {
            encode_channel(re, side)?;
        }

        // §4.2.4 — per-frame LBRR flag sets.
        let encode_per_frame = |re: &mut crate::range_encoder::RangeEncoder,
                                global: bool,
                                bits: u8|
         -> Result<(), Error> {
            if n == 1 {
                // No symbol: the global flag IS the per-frame flag.
                if bits != u8::from(global) {
                    return Err(Error::MalformedPacket);
                }
                return Ok(());
            }
            if !global {
                if bits != 0 {
                    return Err(Error::MalformedPacket);
                }
                return Ok(());
            }
            if bits == 0 || bits & !mask != 0 {
                // Table 4 excludes 0 (the global flag guarantees at
                // least one frame has LBRR).
                return Err(Error::MalformedPacket);
            }
            re.enc_icdf((bits - 1) as usize, per_frame_lbrr_pdf(n), 8);
            Ok(())
        };
        encode_per_frame(re, self.mid.lbrr_flag, self.per_frame_lbrr.mid)?;
        match &self.side {
            Some(side) => encode_per_frame(re, side.lbrr_flag, self.per_frame_lbrr.side)?,
            None => {
                if self.per_frame_lbrr.side != 0 {
                    return Err(Error::MalformedPacket);
                }
            }
        }
        Ok(())
    }

    /// Whether SILK frame `idx` in the mid channel was coded as
    /// active speech (its §4.2.3 VAD flag is set).
    ///
    /// Out-of-range `idx` returns `false`.
    pub fn mid_vad(&self, idx: u8) -> bool {
        idx < self.num_silk_frames && (self.mid.vad_flags >> idx) & 1 == 1
    }

    /// Whether SILK frame `idx` in the side channel was coded as
    /// active speech. Mono frames always return `false`; out-of-range
    /// `idx` returns `false`.
    pub fn side_vad(&self, idx: u8) -> bool {
        match self.side {
            Some(s) => idx < self.num_silk_frames && (s.vad_flags >> idx) & 1 == 1,
            None => false,
        }
    }

    /// Whether SILK frame `idx` of the mid channel has an LBRR
    /// redundancy copy in the bitstream. Out-of-range `idx` returns
    /// `false`.
    pub fn mid_has_lbrr(&self, idx: u8) -> bool {
        idx < self.num_silk_frames && (self.per_frame_lbrr.mid >> idx) & 1 == 1
    }

    /// Whether SILK frame `idx` of the side channel has an LBRR
    /// redundancy copy. Mono frames always return `false`;
    /// out-of-range `idx` returns `false`.
    pub fn side_has_lbrr(&self, idx: u8) -> bool {
        self.side.is_some()
            && idx < self.num_silk_frames
            && (self.per_frame_lbrr.side >> idx) & 1 == 1
    }
}

/// Decode one channel's §4.2.3 VAD bits + LBRR flag from `rd`.
///
/// All bits are uniform binary, so each is one `dec_bit_logp(1)` call
/// (logp = 1 ⇒ probability `2^-1 = 1/2`).
fn decode_channel_header(rd: &mut RangeDecoder<'_>, num_silk_frames: u8) -> SilkChannelHeader {
    let mut vad_flags: u8 = 0;
    for i in 0..num_silk_frames {
        // §4.2.3: "one VAD bit per frame (up to 3), followed by a
        // single flag indicating the presence of LBRR frames".
        let bit = rd.dec_bit_logp(1) as u8;
        // Bit i of vad_flags = VAD flag of SILK frame i. The §4.2.3
        // bit order matches the SILK-frame index because the spec
        // describes "one VAD bit per frame": the i-th bit decoded
        // corresponds to the i-th SILK frame.
        vad_flags |= bit << i;
    }
    let lbrr_flag = rd.dec_bit_logp(1) != 0;
    SilkChannelHeader {
        vad_flags,
        lbrr_flag,
    }
}

/// Decode the §4.2.4 per-frame LBRR flag set for one channel of an
/// Opus frame that carries 2 or 3 SILK frames, returning the packed
/// bitmap. The caller guarantees `num_silk_frames ∈ {2, 3}`.
///
/// The decoded value is in `1..=2^N - 1` (`N = num_silk_frames`) per
/// §4.2.4: "the resulting 2- or 3-bit integer contains the
/// corresponding LBRR flag for each frame, packed in order from the
/// LSB to the MSB". The PDF deliberately excludes value 0 because
/// the channel only reaches this path with its §4.2.3 global LBRR
/// flag set, so at least one of the N flags must be 1.
fn decode_per_frame_lbrr(rd: &mut RangeDecoder<'_>, num_silk_frames: u8) -> u8 {
    let icdf = per_frame_lbrr_pdf(num_silk_frames);
    // Table 4 PDFs both have a zero entry at index 0. Per §4.1.3.3,
    // we drop the leading zero entries and add a constant offset
    // equal to the number of dropped entries (1 here) to the decoded
    // symbol. The truncated iCDF tables live in
    // [`PER_FRAME_LBRR_40MS_ICDF`] and [`PER_FRAME_LBRR_60MS_ICDF`].
    let k = rd.dec_icdf(icdf, 8);
    // k ∈ 0..N-1 after offsetting → return value ∈ 1..=N.
    (k + 1) as u8
}

/// Return the Table 4 iCDF (post-leading-zero truncation; see
/// §4.1.3.3 "drop the entries for any initial zero-probability values
/// and add the constant offset of the first value with a non-zero
/// probability to its return value") for the §4.2.4 per-frame LBRR
/// PDF that matches a given SILK frame count.
///
/// Returns `&[]` for `num_silk_frames` outside the supported set
/// `{2, 3}` (10 / 20 ms Opus frames don't decode any per-frame LBRR
/// symbol).
pub fn per_frame_lbrr_pdf(num_silk_frames: u8) -> &'static [u8] {
    match num_silk_frames {
        2 => PER_FRAME_LBRR_40MS_ICDF,
        3 => PER_FRAME_LBRR_60MS_ICDF,
        _ => &[],
    }
}

/// Table 4 — 40 ms per-frame LBRR PDF, `{0, 53, 53, 150}/256`.
///
/// Cumulative `fh = [0, 53, 106, 256]` (the leading zero is dropped
/// per §4.1.3.3). After the drop the symbol space is `{0, 1, 2}` and
/// the caller adds an offset of 1 to recover the spec's `{1, 2, 3}`
/// (the four possible LBRR bitmaps for a 2-SILK-frame Opus frame:
/// `0b01`, `0b10`, `0b11` — with `0b00` excluded because the global
/// LBRR flag implies at least one LBRR-coded SILK frame).
///
/// iCDF = `256 - fh[k]` for the post-truncation entries plus a
/// terminating 0 = `[203, 150, 0]`. ftb = 8.
pub(crate) const PER_FRAME_LBRR_40MS_ICDF: &[u8] = &[203, 150, 0];

/// Table 4 — 60 ms per-frame LBRR PDF,
/// `{0, 41, 20, 29, 41, 15, 28, 82}/256`.
///
/// Cumulative `fh = [0, 41, 61, 90, 131, 146, 174, 256]`. After the
/// §4.1.3.3 leading-zero drop the symbol space is `{0..6}`; the
/// caller adds an offset of 1 to land in `{1..7}` (the seven
/// possible LBRR bitmaps for a 3-SILK-frame Opus frame).
///
/// iCDF = `[215, 195, 166, 125, 110, 82, 0]`. ftb = 8.
pub(crate) const PER_FRAME_LBRR_60MS_ICDF: &[u8] = &[215, 195, 166, 125, 110, 82, 0];

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // PDF / table transcription self-checks.
    // ----------------------------------------------------------------

    #[test]
    fn table4_40ms_pdf_truncation_is_correct() {
        // Table 4 row 1 — 40 ms — {0, 53, 53, 150}/256.
        let pdf = [0u32, 53, 53, 150];
        assert_eq!(pdf.iter().sum::<u32>(), 256);
        // Cumulative fh, post-truncation: [53, 106, 256].
        let fh: [u32; 3] = [53, 106, 256];
        let derived: [u8; 3] = [
            (256 - fh[0]) as u8,
            (256 - fh[1]) as u8,
            (256 - fh[2]) as u8,
        ];
        assert_eq!(derived, [203u8, 150, 0]);
        assert_eq!(PER_FRAME_LBRR_40MS_ICDF, &derived);
        // Strictly decreasing + terminator zero.
        for w in PER_FRAME_LBRR_40MS_ICDF.windows(2) {
            assert!(w[0] > w[1]);
        }
        assert_eq!(*PER_FRAME_LBRR_40MS_ICDF.last().unwrap(), 0);
    }

    #[test]
    fn table4_60ms_pdf_truncation_is_correct() {
        // Table 4 row 2 — 60 ms — {0, 41, 20, 29, 41, 15, 28, 82}/256.
        let pdf = [0u32, 41, 20, 29, 41, 15, 28, 82];
        assert_eq!(pdf.iter().sum::<u32>(), 256);
        let fh: [u32; 7] = [41, 61, 90, 131, 146, 174, 256];
        let derived: [u8; 7] = [
            (256 - fh[0]) as u8,
            (256 - fh[1]) as u8,
            (256 - fh[2]) as u8,
            (256 - fh[3]) as u8,
            (256 - fh[4]) as u8,
            (256 - fh[5]) as u8,
            (256 - fh[6]) as u8,
        ];
        assert_eq!(derived, [215u8, 195, 166, 125, 110, 82, 0]);
        assert_eq!(PER_FRAME_LBRR_60MS_ICDF, &derived);
        for w in PER_FRAME_LBRR_60MS_ICDF.windows(2) {
            assert!(w[0] > w[1]);
        }
        assert_eq!(*PER_FRAME_LBRR_60MS_ICDF.last().unwrap(), 0);
    }

    #[test]
    fn per_frame_lbrr_pdf_dispatch() {
        assert_eq!(per_frame_lbrr_pdf(2), PER_FRAME_LBRR_40MS_ICDF);
        assert_eq!(per_frame_lbrr_pdf(3), PER_FRAME_LBRR_60MS_ICDF);
        // 1 and out-of-range yield empty.
        assert_eq!(per_frame_lbrr_pdf(1), &[] as &[u8]);
        assert_eq!(per_frame_lbrr_pdf(0), &[] as &[u8]);
        assert_eq!(per_frame_lbrr_pdf(4), &[] as &[u8]);
    }

    // ----------------------------------------------------------------
    // §4.2.2 frame-count dispatch.
    // ----------------------------------------------------------------

    #[test]
    fn silk_frame_count_dispatch_matches_section_4_2_2() {
        // 10 ms and 20 ms ⇒ 1 SILK frame.
        assert_eq!(silk_frame_count(100), Some(1));
        assert_eq!(silk_frame_count(200), Some(1));
        // 40 ms ⇒ 2 SILK frames.
        assert_eq!(silk_frame_count(400), Some(2));
        // 60 ms ⇒ 3 SILK frames.
        assert_eq!(silk_frame_count(600), Some(3));
        // CELT-only frame sizes have no SILK layer.
        assert_eq!(silk_frame_count(25), None);
        assert_eq!(silk_frame_count(50), None);
        // Anything else is not a valid TOC Table 2 frame size.
        assert_eq!(silk_frame_count(0), None);
        assert_eq!(silk_frame_count(1234), None);
    }

    // ----------------------------------------------------------------
    // SilkHeaderBits::decode happy-path bit ordering.
    // ----------------------------------------------------------------

    /// A 10 ms mono frame carries exactly TWO header bits: one VAD
    /// flag + one LBRR flag. The §4.2.3 binary symbols are uniform
    /// `{1, 1}/2` (logp = 1).
    #[test]
    fn decode_mono_10ms_consumes_two_bits() {
        // A 16-byte buffer is far more than enough to keep the range
        // decoder happy. Bit content doesn't matter — both possible
        // decodes are valid header bits.
        let buf = [0x55u8; 16];
        let mut rd = RangeDecoder::new(&buf);
        let tell0 = rd.tell();
        let h = SilkHeaderBits::decode(&mut rd, 1, false).expect("decode");
        let tell1 = rd.tell();
        // 1 VAD bit + 1 LBRR bit.
        assert_eq!(tell1 - tell0, 2);
        assert_eq!(h.num_silk_frames, 1);
        assert!(h.side.is_none());
        // mid VAD bitmap only the low bit is meaningful.
        assert!(h.mid.vad_flags <= 1);
    }

    /// A 60 ms stereo frame with both LBRR flags set should consume:
    /// 3 (mid VAD) + 1 (mid LBRR) + 3 (side VAD) + 1 (side LBRR) +
    /// the 8-bit per-frame LBRR (mid) + the 8-bit per-frame LBRR
    /// (side) header symbols. That's 8 uniform bits plus two
    /// 60 ms-PDF symbols.
    #[test]
    fn decode_stereo_60ms_path_smokes() {
        let buf = [0xFFu8; 32];
        let mut rd = RangeDecoder::new(&buf);
        let h = SilkHeaderBits::decode(&mut rd, 3, true).expect("decode");
        assert_eq!(h.num_silk_frames, 3);
        assert!(h.side.is_some());
        // Only the low 3 bits of each VAD bitmap are meaningful.
        assert!(h.mid.vad_flags < 8, "mid VAD {:b}", h.mid.vad_flags);
        assert!(
            h.side.unwrap().vad_flags < 8,
            "side VAD {:b}",
            h.side.unwrap().vad_flags
        );
        // Only the low 3 bits of each per-frame LBRR bitmap are
        // meaningful.
        assert!(h.per_frame_lbrr.mid < 8);
        assert!(h.per_frame_lbrr.side < 8);
    }

    /// `num_silk_frames` out of `{1, 2, 3}` is a packet-level error.
    #[test]
    fn decode_rejects_invalid_silk_frame_count() {
        let buf = [0u8; 4];
        let mut rd = RangeDecoder::new(&buf);
        assert_eq!(
            SilkHeaderBits::decode(&mut rd, 0, false),
            Err(Error::MalformedPacket)
        );
        let mut rd = RangeDecoder::new(&buf);
        assert_eq!(
            SilkHeaderBits::decode(&mut rd, 4, false),
            Err(Error::MalformedPacket)
        );
    }

    /// For Opus frames ≤ 20 ms (`num_silk_frames == 1`), §4.2.3
    /// states the global LBRR flag already encodes the presence of
    /// the single LBRR frame: the per-frame LBRR bitmap mirrors the
    /// header flag without consuming any extra bits.
    #[test]
    fn decode_10ms_per_frame_lbrr_mirrors_global_flag_no_extra_bits() {
        // Construct a frame where the global LBRR flag is 1. The
        // §4.2.3 dec_bit_logp(1) decoder treats `val < s = rng>>1`
        // as a "1": picking a buffer whose initial `val = 127 -
        // (b0>>1)` falls in [0, 64) does so. The decoder is
        // initialised with `b0 = buf[0]`, val = 127 - (b0>>1). To get
        // val in [0, 64) we need b0>>1 >= 64 ⇒ b0 >= 128. We use 0xFF
        // ⇒ val = 127 - 127 = 0, which is "1".
        let buf = [0xFFu8; 8];
        let mut rd = RangeDecoder::new(&buf);
        let tell0 = rd.tell();
        let h = SilkHeaderBits::decode(&mut rd, 1, false).expect("decode");
        let tell1 = rd.tell();
        // Two uniform bits total — no per-frame LBRR symbol decoded.
        assert_eq!(tell1 - tell0, 2);
        // mid LBRR flag should be 1 with this buffer.
        assert!(h.mid.lbrr_flag, "mid LBRR flag should be 1");
        // Per-frame mid LBRR bitmap mirrors the header flag.
        assert_eq!(h.per_frame_lbrr.mid, 1);
        assert_eq!(h.per_frame_lbrr.side, 0);
        // Accessor agrees.
        assert!(h.mid_has_lbrr(0));
        assert!(!h.mid_has_lbrr(1));
        assert!(!h.side_has_lbrr(0));
    }

    /// Channels without their global LBRR flag set must skip the
    /// per-frame LBRR symbol entirely (mid OR side), even on a 60 ms
    /// Opus frame.
    #[test]
    fn decode_60ms_skips_per_frame_lbrr_when_global_flag_unset() {
        // Buffer engineered so that decoded VAD/LBRR bits are all 0:
        // val = 127 - (b0>>1) is the initial discriminator, and
        // s = rng >> 1 = 64. For "0", we need val >= s = 64, so we
        // want b0 such that 127 - (b0>>1) >= 64 ⇒ b0 <= 126.
        let buf = [0x00u8; 16];
        let mut rd = RangeDecoder::new(&buf);
        let tell0 = rd.tell();
        let h = SilkHeaderBits::decode(&mut rd, 3, true).expect("decode");
        let tell1 = rd.tell();
        // All 8 header bits should decode to "0" → both LBRR flags
        // unset → no per-frame LBRR symbol consumed.
        // 3 mid VAD + 1 mid LBRR + 3 side VAD + 1 side LBRR = 8 bits.
        assert_eq!(tell1 - tell0, 8);
        assert!(!h.mid.lbrr_flag);
        assert!(!h.side.unwrap().lbrr_flag);
        assert_eq!(h.per_frame_lbrr.mid, 0);
        assert_eq!(h.per_frame_lbrr.side, 0);
        assert_eq!(h.mid.vad_flags, 0);
        assert_eq!(h.side.unwrap().vad_flags, 0);
    }

    // ----------------------------------------------------------------
    // VAD / LBRR accessor surface.
    // ----------------------------------------------------------------

    #[test]
    fn vad_accessors_match_bitmap_for_mid_and_side() {
        let h = SilkHeaderBits {
            num_silk_frames: 3,
            mid: SilkChannelHeader {
                vad_flags: 0b101,
                lbrr_flag: true,
            },
            side: Some(SilkChannelHeader {
                vad_flags: 0b010,
                lbrr_flag: false,
            }),
            per_frame_lbrr: PerFrameLbrr {
                mid: 0b110,
                side: 0,
            },
        };
        assert!(h.mid_vad(0));
        assert!(!h.mid_vad(1));
        assert!(h.mid_vad(2));
        // Beyond num_silk_frames returns false.
        assert!(!h.mid_vad(3));
        assert!(!h.side_vad(0));
        assert!(h.side_vad(1));
        assert!(!h.side_vad(2));
        // Per-frame LBRR accessor.
        assert!(!h.mid_has_lbrr(0));
        assert!(h.mid_has_lbrr(1));
        assert!(h.mid_has_lbrr(2));
        assert!(!h.side_has_lbrr(0));
        assert!(!h.side_has_lbrr(1));
        assert!(!h.side_has_lbrr(2));
    }

    #[test]
    fn vad_accessors_zero_for_missing_side_channel() {
        let h = SilkHeaderBits {
            num_silk_frames: 1,
            mid: SilkChannelHeader {
                vad_flags: 1,
                lbrr_flag: true,
            },
            side: None,
            per_frame_lbrr: PerFrameLbrr { mid: 1, side: 0 },
        };
        assert!(!h.side_vad(0));
        assert!(!h.side_has_lbrr(0));
    }

    // ----------------------------------------------------------------
    // §4.2.4 per-frame LBRR symbol exhaustion.
    // ----------------------------------------------------------------

    /// `decode_per_frame_lbrr` for 40 ms should never return 0 (the
    /// truncated PDF excludes the value 0 → offset 1 places the
    /// result in `{1, 2, 3}` for every possible decoder state).
    #[test]
    fn decode_per_frame_lbrr_40ms_never_returns_zero() {
        for b0 in 0u16..=255 {
            for b1 in [0x00u8, 0xFFu8, 0x55, 0xAA, 0x33] {
                let buf = [b0 as u8, b1, 0x11, 0x22, 0x44, 0x88, 0x10, 0x20];
                let mut rd = RangeDecoder::new(&buf);
                let v = decode_per_frame_lbrr(&mut rd, 2);
                assert!((1..=3).contains(&v), "40ms LBRR {v} out of 1..=3");
            }
        }
    }

    /// `decode_per_frame_lbrr` for 60 ms returns `{1..=7}`.
    #[test]
    fn decode_per_frame_lbrr_60ms_in_range() {
        for b0 in 0u16..=255 {
            for b1 in [0x00u8, 0xFFu8, 0x55, 0xAA, 0x33] {
                let buf = [b0 as u8, b1, 0x11, 0x22, 0x44, 0x88, 0x10, 0x20];
                let mut rd = RangeDecoder::new(&buf);
                let v = decode_per_frame_lbrr(&mut rd, 3);
                assert!((1..=7).contains(&v), "60ms LBRR {v} out of 1..=7");
            }
        }
    }

    /// Sweeping a variety of buffers should produce all seven
    /// possible 60 ms LBRR bitmaps (1..=7) at least once.
    #[test]
    fn decode_per_frame_lbrr_60ms_covers_full_symbol_set() {
        let mut seen = [false; 8];
        for b0 in 0u16..=255 {
            let buf = [
                b0 as u8,
                (b0 as u8) ^ 0x5A,
                0xC3,
                0x3C,
                0x96,
                0x69,
                0xAA,
                0x55,
            ];
            let mut rd = RangeDecoder::new(&buf);
            let v = decode_per_frame_lbrr(&mut rd, 3);
            seen[v as usize] = true;
        }
        // Every value 1..=7 must show up. Value 0 must never.
        assert!(!seen[0], "value 0 should be unreachable");
        for (v, hit) in seen.iter().enumerate().skip(1) {
            assert!(*hit, "value {v} never decoded in 60ms LBRR sweep");
        }
    }
}
