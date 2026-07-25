//! Codec-id → family mapping and the length-prefixed → Annex B NAL reframing the demuxer
//! needs for the ISO-BMFF-style video codec mappings (spec: `spec/MATROSKA.md`; RFC 9559 §12
//! codec mappings; the Matroska codec registry at `codec.mkvtoolnix.download`).
//!
//! ## Two shapes of video track
//! - **WebM codecs** — `V_VP8`, `V_VP9`, `V_AV1` — store frames *raw*: one Block is exactly
//!   one codec frame / temporal unit, with no CodecPrivate. The demuxer forwards those bytes
//!   verbatim (naming only — see [`family_for`]).
//! - **ISO-BMFF codecs** — `V_MPEG4/ISO/AVC` (H.264) and `V_MPEGH/ISO/HEVC` (H.265) — store
//!   frames as **length-prefixed NAL units** and carry the parameter sets in the CodecPrivate
//!   as an `AVCDecoderConfigurationRecord` / `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15).
//!   Our H.264/H.265 decoders consume **Annex B** (`00 00 00 01` start-code) elementary
//!   streams, so the demuxer must reframe: emit the parameter sets as an Annex B head once,
//!   then convert each Block's length-prefixed NALs to start-code form (one access unit per
//!   output buffer).
//!
//! ## Robustness (spec: "a crash on bad input is a P0")
//! CodecPrivate and Block bytes are untrusted. Every field here is bounds-checked and every
//! length is validated against the remaining input; a malformed record or block yields an
//! [`Err`], never a panic or an out-of-range slice. The demuxer warns-and-drops on such an
//! error rather than killing the pipeline.

use streamcraft_core::format::OfferDesc;

/// The demuxer announce family for a video (or audio) track, keyed by CodecID (RFC 9559 §12
/// codec mappings). The families match the sink offers of the streamcraft decoders so a
/// dynamic pad linking `mkvdemux.video_N ! vp8dec.sink` negotiates:
/// - `V_VP8` → `vp8`, `V_VP9` → `vp9`, `V_AV1` → `av1`;
/// - `V_MPEG4/ISO/AVC` → `h264/annexb`, `V_MPEGH/ISO/HEVC` → `h265/annexb`;
/// - `A_FLAC` → `flac`; anything else → `bytes` (the unknown-codec fallback).
///
/// The families are `&'static str` so they can seed a pad's static offer menu (a pad may only
/// announce a family it already offered — the vocabulary is interned from those offers).
pub fn family_for(codec_id: &str) -> &'static str {
    match codec_id {
        "A_FLAC" => "flac",
        "V_VP8" => "vp8",
        "V_VP9" => "vp9",
        "V_AV1" => "av1",
        "V_MPEG4/ISO/AVC" => "h264/annexb",
        "V_MPEGH/ISO/HEVC" => "h265/annexb",
        _ => "bytes",
    }
}

/// The offer menu for a track's dynamic src pad: exactly [`family_for`]'s announce
/// family, plus the `bytes` escape so a generic byte sink can still tap the track.
/// **Per-track menus are what make negotiation-driven autoplug select**: one shared
/// all-families menu let `h265dec` link an h264 track (link-time intersection admits
/// any family on the menu; the mismatch only surfaced at runtime as every access unit
/// dropping). Unconstrained `any` offers — the concrete width/height ride the runtime
/// announcement, and their field names are interned by the consumer's offers.
pub fn offers_for(codec_id: &str) -> &'static [OfferDesc] {
    static FLAC: [OfferDesc; 2] = [OfferDesc::any("flac"), OfferDesc::any("bytes")];
    static VP8: [OfferDesc; 2] = [OfferDesc::any("vp8"), OfferDesc::any("bytes")];
    static VP9: [OfferDesc; 2] = [OfferDesc::any("vp9"), OfferDesc::any("bytes")];
    static AV1: [OfferDesc; 2] = [OfferDesc::any("av1"), OfferDesc::any("bytes")];
    static H264: [OfferDesc; 2] = [OfferDesc::any("h264/annexb"), OfferDesc::any("bytes")];
    static H265: [OfferDesc; 2] = [OfferDesc::any("h265/annexb"), OfferDesc::any("bytes")];
    static BYTES: [OfferDesc; 1] = [OfferDesc::any("bytes")];
    match codec_id {
        "A_FLAC" => &FLAC,
        "V_VP8" => &VP8,
        "V_VP9" => &VP9,
        "V_AV1" => &AV1,
        "V_MPEG4/ISO/AVC" => &H264,
        "V_MPEGH/ISO/HEVC" => &H265,
        _ => &BYTES,
    }
}

/// The 4-byte Annex B start code prepended to every NAL unit (ITU-T H.264 / H.265 Annex B).
/// We always emit the 4-byte form (`00 00 00 01`); the 3-byte form is also legal but the
/// 4-byte one is unambiguous and what our decoders' test fixtures use.
const START_CODE: [u8; 4] = [0x00, 0x00, 0x00, 0x01];

/// A parse failure reframing an AVC/HEVC configuration record or a length-prefixed block.
/// Distinct from the reader's `ReadError` because this is a codec-layer concern (the demuxer
/// maps it to a bus warning-and-drop, not a stream-fatal error).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ReframeError {
    /// The configuration record or a block ran out of bytes before a declared field/NAL.
    Truncated(&'static str),
    /// A structurally impossible value (e.g. a NAL length that overflows the buffer).
    Malformed(&'static str),
}

/// How a track's frames and its stream-start head are reframed on the way out of the demuxer.
/// Chosen once from the CodecID (via [`for_codec`](Reframer::for_codec)); applied per frame.
#[derive(Clone, Debug)]
pub enum Reframer {
    /// Frames pass through verbatim (WebM video, and any raw-`bytes` track). The head is
    /// whatever the demuxer already computed (empty for video, the FLAC head for `A_FLAC`).
    Passthrough,
    /// ISO-BMFF NAL video: `length_size` bytes precede each NAL in a Block; convert both the
    /// CodecPrivate parameter sets and each block to Annex B start-code form.
    Nal { length_size: usize },
}

impl Reframer {
    /// The reframer for a CodecID. `V_MPEG4/ISO/AVC` and `V_MPEGH/ISO/HEVC` reframe NALs; every
    /// other codec passes through (raw WebM frames, native FLAC frames, unknown bytes).
    pub fn for_codec(codec_id: &str) -> Reframer {
        match codec_id {
            "V_MPEG4/ISO/AVC" | "V_MPEGH/ISO/HEVC" => Reframer::Nal { length_size: 4 },
            _ => Reframer::Passthrough,
        }
    }

    /// Reframe one Block's payload to the bytes emitted downstream. For [`Passthrough`] this is
    /// the input unchanged (borrowed); for [`Nal`] each length-prefixed NAL becomes a
    /// start-code NAL (owned). Errors on a malformed length-prefixed block (never panics).
    ///
    /// [`Passthrough`]: Reframer::Passthrough
    /// [`Nal`]: Reframer::Nal
    pub fn reframe_block<'a>(&self, block: &'a [u8]) -> Result<std::borrow::Cow<'a, [u8]>, ReframeError> {
        match self {
            Reframer::Passthrough => Ok(std::borrow::Cow::Borrowed(block)),
            Reframer::Nal { length_size } => {
                Ok(std::borrow::Cow::Owned(length_prefixed_to_annex_b(block, *length_size)?))
            }
        }
    }
}

/// The Annex B stream-start head for an ISO-BMFF NAL track from its CodecPrivate: the parameter
/// sets (SPS/PPS for AVC, VPS/SPS/PPS for HEVC) as start-code NALs, and the NAL length size the
/// blocks use. Emitted once, before the first frame, so the downstream decoder sees the
/// parameter sets it needs. Returns `(annex_b_head, length_size)`.
///
/// `is_hevc` selects the record layout: an `AVCDecoderConfigurationRecord` (ISO/IEC 14496-15
/// §5.3.3.1) for false, an `HEVCDecoderConfigurationRecord` (§8.3.3.1) for true.
pub fn nal_head_from_config(codec_private: &[u8], is_hevc: bool) -> Result<(Vec<u8>, usize), ReframeError> {
    if is_hevc {
        parse_hvcc(codec_private)
    } else {
        parse_avcc(codec_private)
    }
}

/// Parse an `AVCDecoderConfigurationRecord` (ISO/IEC 14496-15 §5.3.3.1) into its Annex B
/// parameter-set head and the NAL length size. Layout:
///
/// ```text
/// configurationVersion        u8   (== 1)
/// AVCProfileIndication        u8
/// profile_compatibility       u8
/// AVCLevelIndication          u8
/// 111111 | lengthSizeMinusOne u8   (low 2 bits = lengthSize - 1)
/// 111 | numOfSPS              u8   (low 5 bits)
///   [ SPS length u16, SPS bytes ] * numOfSPS
/// numOfPPS                    u8
///   [ PPS length u16, PPS bytes ] * numOfPPS
/// ```
fn parse_avcc(data: &[u8]) -> Result<(Vec<u8>, usize), ReframeError> {
    let mut r = Cursor::new(data);
    let version = r.u8("avcC configurationVersion")?;
    if version != 1 {
        return Err(ReframeError::Malformed("avcC configurationVersion != 1"));
    }
    r.skip(3, "avcC profile/compat/level")?; // profile, compatibility, level
    let length_size = (r.u8("avcC lengthSizeMinusOne")? & 0x03) as usize + 1;
    let num_sps = r.u8("avcC numOfSequenceParameterSets")? & 0x1F;
    let mut head = Vec::new();
    for _ in 0..num_sps {
        append_nal(&mut head, r.length_prefixed_u16("avcC SPS")?);
    }
    let num_pps = r.u8("avcC numOfPictureParameterSets")?;
    for _ in 0..num_pps {
        append_nal(&mut head, r.length_prefixed_u16("avcC PPS")?);
    }
    Ok((head, length_size))
}

/// Parse an `HEVCDecoderConfigurationRecord` (ISO/IEC 14496-15 §8.3.3.1) into its Annex B
/// parameter-set head and the NAL length size. The fixed prefix is 22 octets; then an array of
/// NAL-unit *arrays*, one per NAL type (typically VPS, SPS, PPS):
///
/// ```text
/// configurationVersion        u8   (== 1)
/// 21 fixed octets (profile/tier/level, format flags, lengthSizeMinusOne at octet 21)
/// numOfArrays                 u8
///   per array:
///     array_completeness|reserved|NAL_unit_type  u8
///     numNalus                                    u16
///       [ nalUnitLength u16, NAL bytes ] * numNalus
/// ```
fn parse_hvcc(data: &[u8]) -> Result<(Vec<u8>, usize), ReframeError> {
    let mut r = Cursor::new(data);
    let version = r.u8("hvcC configurationVersion")?;
    if version != 1 {
        return Err(ReframeError::Malformed("hvcC configurationVersion != 1"));
    }
    // Octets 1..21 are profile/tier/level and format info; octet 21's low 2 bits are
    // lengthSizeMinusOne. Read the block, then pick out lengthSizeMinusOne.
    let fixed = r.take(21, "hvcC fixed prefix")?;
    let length_size = (fixed[20] & 0x03) as usize + 1;
    let num_arrays = r.u8("hvcC numOfArrays")?;
    let mut head = Vec::new();
    for _ in 0..num_arrays {
        let _type_byte = r.u8("hvcC array type")?; // array_completeness | reserved | NAL type
        let num_nalus = r.u16("hvcC numNalus")?;
        for _ in 0..num_nalus {
            append_nal(&mut head, r.length_prefixed_u16("hvcC NAL")?);
        }
    }
    Ok((head, length_size))
}

/// Convert a Block of length-prefixed NAL units to an Annex B access unit: each NAL is preceded
/// by a `length_size`-octet big-endian length, replaced by a `00 00 00 01` start code. Errors
/// on a truncated length field or a NAL length that runs past the block (never panics).
fn length_prefixed_to_annex_b(block: &[u8], length_size: usize) -> Result<Vec<u8>, ReframeError> {
    debug_assert!((1..=4).contains(&length_size));
    let mut out = Vec::with_capacity(block.len() + START_CODE.len());
    let mut at = 0usize;
    while at < block.len() {
        // Read the big-endian NAL length.
        let end = at
            .checked_add(length_size)
            .ok_or(ReframeError::Malformed("NAL length offset overflow"))?;
        let len_bytes = block.get(at..end).ok_or(ReframeError::Truncated("NAL length field"))?;
        let mut nal_len = 0usize;
        for &b in len_bytes {
            nal_len = (nal_len << 8) | b as usize;
        }
        at = end;
        let nal_end = at
            .checked_add(nal_len)
            .ok_or(ReframeError::Malformed("NAL length overflow"))?;
        let nal = block.get(at..nal_end).ok_or(ReframeError::Truncated("NAL unit runs past the block"))?;
        // A zero-length NAL is degenerate but not fatal — skip it rather than emit a bare start
        // code (some muxers pad; the decoder would ignore an empty NAL anyway).
        if !nal.is_empty() {
            append_nal(&mut out, nal);
        }
        at = nal_end;
    }
    Ok(out)
}

/// Append one NAL unit in Annex B form: the 4-octet start code, then the NAL bytes.
fn append_nal(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&START_CODE);
    out.extend_from_slice(nal);
}

/// A minimal bounds-checked forward cursor over the configuration-record bytes — every read is
/// fallible so a truncated record errors rather than panicking (spec: untrusted input).
struct Cursor<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, ReframeError> {
        let b = *self.data.get(self.at).ok_or(ReframeError::Truncated(what))?;
        self.at += 1;
        Ok(b)
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, ReframeError> {
        let hi = self.u8(what)? as u16;
        let lo = self.u8(what)? as u16;
        Ok((hi << 8) | lo)
    }

    fn skip(&mut self, n: usize, what: &'static str) -> Result<(), ReframeError> {
        let _ = self.take(n, what)?;
        Ok(())
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], ReframeError> {
        let end = self.at.checked_add(n).ok_or(ReframeError::Malformed(what))?;
        let s = self.data.get(self.at..end).ok_or(ReframeError::Truncated(what))?;
        self.at = end;
        Ok(s)
    }

    /// Read a u16 length, then that many NAL bytes — the SPS/PPS/NAL entry shape shared by
    /// both configuration records.
    fn length_prefixed_u16(&mut self, what: &'static str) -> Result<&'a [u8], ReframeError> {
        let len = self.u16(what)? as usize;
        self.take(len, what)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_mapping_covers_the_codec_ids() {
        assert_eq!(family_for("V_VP8"), "vp8");
        assert_eq!(family_for("V_VP9"), "vp9");
        assert_eq!(family_for("V_AV1"), "av1");
        assert_eq!(family_for("V_MPEG4/ISO/AVC"), "h264/annexb");
        assert_eq!(family_for("V_MPEGH/ISO/HEVC"), "h265/annexb");
        assert_eq!(family_for("A_FLAC"), "flac");
        assert_eq!(family_for("A_OPUS"), "bytes", "unknown → bytes fallback");
        assert_eq!(family_for(""), "bytes", "empty codec id → bytes fallback");
    }

    #[test]
    fn only_nal_codecs_reframe() {
        assert!(matches!(Reframer::for_codec("V_MPEG4/ISO/AVC"), Reframer::Nal { length_size: 4 }));
        assert!(matches!(Reframer::for_codec("V_MPEGH/ISO/HEVC"), Reframer::Nal { length_size: 4 }));
        assert!(matches!(Reframer::for_codec("V_VP8"), Reframer::Passthrough));
        assert!(matches!(Reframer::for_codec("A_FLAC"), Reframer::Passthrough));
    }

    /// A hand-built avcC (one SPS, one PPS, lengthSize 4) yields SPS+PPS as Annex B and a
    /// length size of 4.
    #[test]
    fn avcc_parses_to_annex_b_head() {
        let sps = [0x67, 0x42, 0x00, 0x0A];
        let pps = [0x68, 0xCE, 0x3C, 0x80];
        let mut cfg = vec![
            1, // configurationVersion
            0x42, 0x00, 0x0A, // profile, compat, level
            0xFF, // 111111 | lengthSizeMinusOne=3 → lengthSize 4
            0xE1, // 111 | numOfSPS=1
        ];
        cfg.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        cfg.extend_from_slice(&sps);
        cfg.push(1); // numOfPPS
        cfg.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        cfg.extend_from_slice(&pps);

        let (head, length_size) = parse_avcc(&cfg).expect("avcC parses");
        assert_eq!(length_size, 4);
        let mut want = Vec::new();
        want.extend_from_slice(&START_CODE);
        want.extend_from_slice(&sps);
        want.extend_from_slice(&START_CODE);
        want.extend_from_slice(&pps);
        assert_eq!(head, want, "SPS then PPS as 4-byte start-code NALs");
    }

    /// A hand-built hvcC (three arrays: VPS, SPS, PPS) yields VPS+SPS+PPS as Annex B, length
    /// size 4.
    #[test]
    fn hvcc_parses_to_annex_b_head() {
        let vps = [0x40, 0x01, 0x0C];
        let sps = [0x42, 0x01, 0x01];
        let pps = [0x44, 0x01, 0xC0];
        let mut cfg = vec![1u8]; // configurationVersion
        cfg.extend_from_slice(&[0u8; 20]); // 20 octets of profile/tier/level/format info
        cfg.push(0xFC | 0x03); // octet 21: lengthSizeMinusOne=3 in the low 2 bits
        cfg.push(3); // numOfArrays
        for (nal_type, nal) in [(32u8, &vps[..]), (33, &sps[..]), (34, &pps[..])] {
            cfg.push(nal_type); // array_completeness|reserved|NAL_unit_type
            cfg.extend_from_slice(&1u16.to_be_bytes()); // numNalus = 1
            cfg.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            cfg.extend_from_slice(nal);
        }
        let (head, length_size) = parse_hvcc(&cfg).expect("hvcC parses");
        assert_eq!(length_size, 4);
        let mut want = Vec::new();
        for nal in [&vps[..], &sps[..], &pps[..]] {
            want.extend_from_slice(&START_CODE);
            want.extend_from_slice(nal);
        }
        assert_eq!(head, want, "VPS, SPS, PPS as 4-byte start-code NALs");
    }

    /// Length-prefixed NALs (4-byte length) reframe to Annex B start-code NALs, one access
    /// unit's worth in one call.
    #[test]
    fn length_prefixed_block_reframes() {
        let n1 = [0x65, 0x11, 0x22];
        let n2 = [0x41, 0x33];
        let mut block = Vec::new();
        block.extend_from_slice(&(n1.len() as u32).to_be_bytes());
        block.extend_from_slice(&n1);
        block.extend_from_slice(&(n2.len() as u32).to_be_bytes());
        block.extend_from_slice(&n2);

        let out = length_prefixed_to_annex_b(&block, 4).expect("reframe");
        let mut want = Vec::new();
        want.extend_from_slice(&START_CODE);
        want.extend_from_slice(&n1);
        want.extend_from_slice(&START_CODE);
        want.extend_from_slice(&n2);
        assert_eq!(out, want);
    }

    /// A 2-byte length size is honoured (some records use it).
    #[test]
    fn length_size_two_reframes() {
        let nal = [0x67, 0xAB];
        let mut block = Vec::new();
        block.extend_from_slice(&(nal.len() as u16).to_be_bytes());
        block.extend_from_slice(&nal);
        let out = length_prefixed_to_annex_b(&block, 2).expect("reframe");
        let mut want = Vec::new();
        want.extend_from_slice(&START_CODE);
        want.extend_from_slice(&nal);
        assert_eq!(out, want);
    }

    // --- Bad-input table: malformed records/blocks error, never panic (spec: P0) ---

    #[test]
    fn malformed_records_error_not_panic() {
        let cases: &[(&str, Vec<u8>, bool)] = &[
            ("empty avcC", vec![], false),
            ("avcC wrong version", vec![2, 0, 0, 0, 0xFF, 0xE0], false),
            ("avcC truncated before SPS count", vec![1, 0x42, 0x00, 0x0A, 0xFF], false),
            (
                "avcC SPS length runs past end",
                vec![1, 0x42, 0x00, 0x0A, 0xFF, 0xE1, 0xFF, 0xFF, 0x00],
                false,
            ),
            ("empty hvcC", vec![], true),
            ("hvcC wrong version", vec![9], true),
            ("hvcC truncated prefix", vec![1, 0, 0, 0], true),
        ];
        for (name, bytes, is_hevc) in cases {
            let r = nal_head_from_config(bytes, *is_hevc);
            assert!(r.is_err(), "{name} must error, not parse");
        }
    }

    #[test]
    fn malformed_blocks_error_not_panic() {
        // Length field claims 0xFF bytes but only 1 present.
        let block = vec![0x00, 0x00, 0x00, 0xFF, 0x41];
        assert!(length_prefixed_to_annex_b(&block, 4).is_err());
        // Truncated length field itself.
        let block = vec![0x00, 0x00];
        assert!(length_prefixed_to_annex_b(&block, 4).is_err());
        // An empty block reframes to empty (no NALs) — valid, not an error.
        assert_eq!(length_prefixed_to_annex_b(&[], 4).unwrap(), Vec::<u8>::new());
    }
}
