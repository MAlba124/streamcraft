//! Format sniffing from the head of a file.
//!
//! Deliberately **independent of `pf_play::probe`**: that typefinder lives in the player
//! crate, which depends on every demuxer and decoder in the workspace. A tag scanner that
//! pulled `pf-play` in to look at eight magic bytes would inherit a dependency graph two
//! orders of magnitude larger than the work it does. Fifty lines and a citation per match
//! arm is the cheaper trade — and the two are allowed to disagree, because they answer
//! different questions ("what pipeline do I build" vs. "whose metadata layout is this").
//!
//! Every arm cites the specification that defines its magic (workspace standing rule:
//! every non-trivial structure cites its source at the point of use).

/// A container/codec family, as far as the first bytes of a file can tell.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Format {
    Flac,
    Mp3,
    Mp4,
    OggOpus,
    OggVorbis,
    OggFlac,
    OggSpeex,
    Wav,
    Aiff,
    Ape,
    WavPack,
    Musepack,
    Mkv,
    #[default]
    Unknown,
}

impl Format {
    /// A short lowercase name, for printing.
    pub fn name(self) -> &'static str {
        match self {
            Format::Flac => "flac",
            Format::Mp3 => "mp3",
            Format::Mp4 => "mp4",
            Format::OggOpus => "ogg/opus",
            Format::OggVorbis => "ogg/vorbis",
            Format::OggFlac => "ogg/flac",
            Format::OggSpeex => "ogg/speex",
            Format::Wav => "wav",
            Format::Aiff => "aiff",
            Format::Ape => "ape",
            Format::WavPack => "wavpack",
            Format::Musepack => "musepack",
            Format::Mkv => "mkv",
            Format::Unknown => "unknown",
        }
    }

    /// Whether this scanner extracts tags and props for the format. Anything unrecognised is
    /// reported with its file facts and empty tags (see the crate docs' coverage table).
    pub fn is_parsed(self) -> bool {
        !matches!(self, Format::Unknown)
    }
}

/// What [`sniff`] concluded.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sniffed {
    pub format: Format,
    /// Offset of the format's own first byte. Non-zero only when an ID3v2 tag precedes the
    /// stream, which is legal in front of MP3 *and* (in the wild) FLAC and others.
    pub body: u64,
}

/// The size of a leading ID3v2 tag, or `None` if `head` does not start with one.
///
/// ID3v2.4.0 structure §3.1: a 10-byte header — `"ID3"`, two version bytes, one flags byte,
/// then four **synchsafe** size bytes (7 significant bits each, high bit always zero, so a
/// tag size can never contain a false frame sync). The size excludes the header, and
/// excludes the 10-byte footer, whose presence is flags bit 4 (`%0001_0000`, §3.1 "Footer
/// present"). `TotalTagSize = 10 + Size (+ 10 when a footer is present)`.
pub(crate) fn id3v2_len(head: &[u8]) -> Option<u64> {
    let h = head.get(..10)?;
    if &h[0..3] != b"ID3" {
        return None;
    }
    // A version byte of 0xFF is invalid (§3.1) — reject rather than trust the size field.
    if h[3] == 0xFF || h[4] == 0xFF {
        return None;
    }
    // Synchsafe: reject any byte with its high bit set instead of silently mis-sizing.
    if h[6..10].iter().any(|b| b & 0x80 != 0) {
        return None;
    }
    let size = h[6..10].iter().fold(0u64, |acc, &b| (acc << 7) | u64::from(b & 0x7F));
    let footer = if h[5] & 0x10 != 0 { 10 } else { 0 };
    Some(10 + size + footer)
}

/// Classify a file from its head. `head` is the prefix actually read (possibly short).
pub(crate) fn sniff(head: &[u8]) -> Sniffed {
    // An ID3v2 tag may sit in front of any stream (ID3v2.4.0 structure §3.1 places it at
    // the very start of the file). Skip it once and re-sniff what follows, so an
    // ID3-prefixed FLAC is a FLAC and not "probably MP3"; if the tag runs past the prefix
    // there is nothing left to look at, and MP3 is the overwhelmingly likely answer.
    //
    // That re-sniff is what makes an ID3v2-prefixed Monkey's Audio, WavPack or Musepack file
    // — all three of which exist in the wild, since the APEv2 family tolerates a leading ID3
    // tag and Monkey's Audio's own SDK skips one before looking for its magic — come out as
    // itself rather than as "probably MP3". Nothing extra is needed for them beyond the
    // magics in [`sniff_bare`]; `body` then carries the offset of the real header, which is
    // what each parser is handed. The MP3 fallback remains for the two cases where there is
    // genuinely nothing to look at: a tag longer than the prefix read, and bytes behind the
    // tag that match no magic at all.
    if let Some(len) = id3v2_len(head) {
        let body = usize::try_from(len).unwrap_or(usize::MAX);
        return match head.get(body..) {
            Some(rest) if !rest.is_empty() => {
                let inner = sniff_bare(rest);
                Sniffed {
                    format: if inner == Format::Unknown { Format::Mp3 } else { inner },
                    body: len,
                }
            }
            _ => Sniffed { format: Format::Mp3, body: len },
        };
    }
    Sniffed { format: sniff_bare(head), body: 0 }
}

/// Magic-number classification with no ID3 handling — the inner half of [`sniff`].
fn sniff_bare(head: &[u8]) -> Format {
    let b = head;
    // RFC 9639 §8: "A FLAC bitstream consists of the fLaC marker at the beginning of the
    // stream", followed by the STREAMINFO metadata block.
    if b.starts_with(b"fLaC") {
        return Format::Flac;
    }
    // RIFF (Multimedia Programming Interface and Data Specifications 1.0, IBM/Microsoft,
    // August 1991, §"RIFF Form"): `"RIFF" <u32 size> <formType>`; the WAVE form is formType
    // `"WAVE"` (§"WAVE Form").
    if b.len() >= 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WAVE" {
        return Format::Wav;
    }
    // ISO/IEC 14496-12 §4.3 (File Type Box): the first box of an ISO base media file is
    // `ftyp`, and a box is `<u32 size><u32 type>` (§4.2), so the type sits at offset 4.
    if b.len() >= 12 && &b[4..8] == b"ftyp" {
        return Format::Mp4;
    }
    // AIFF (Apple, *Audio Interchange File Format* 1.3, "File Structure"): an EA IFF 85
    // `FORM` container whose form type is `AIFF`, or `AIFC` for the 1991 compressed variant
    // (`spec/AIFF.md`). The form type is required — `FORM` alone also introduces IFF-85
    // documents that are not audio (`8SVX`, `ILBM`, `ANIM`).
    if b.len() >= 12 && &b[0..4] == b"FORM" && matches!(&b[8..12], b"AIFF" | b"AIFC") {
        return Format::Aiff;
    }
    // WavPack (*WavPack 4 & 5 Binary File / Block Format* §2.0): every block opens
    // `char ckID [4]; // "wvpk"`.
    if b.starts_with(crate::wv::MAGIC) {
        return Format::WavPack;
    }
    // Monkey's Audio (`APE_COMMON_HEADER::cID`): `'MAC '`, or `'MACF'` for the large-file
    // variant MAC 12.x writes — same layout (`spec/APE.md`).
    if b.starts_with(crate::ape::MAGIC) || b.starts_with(crate::ape::MAGIC_LARGE) {
        return Format::Ape;
    }
    // Musepack (`spec/MPC.md`, "Sniffing"): SV8 is `MPCK`; SV7 is `MP+` followed by a version
    // octet whose **low nibble** is the stream version — the high nibble is the PNS flag, so
    // the octet is `0x07` or `0x17` and must be masked before it is compared.
    if b.starts_with(crate::mpc::MAGIC_SV8) {
        return Format::Musepack;
    }
    if b.starts_with(crate::mpc::MAGIC_SV7) && b.get(3).is_some_and(|v| v & 0x0F == 7) {
        return Format::Musepack;
    }
    // RFC 8794 §11.2.3 / §4: an EBML document begins with the EBML Header element, whose
    // ID is 0x1A45DFA3. Matroska (and WebM) are EBML documents.
    if b.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Format::Mkv;
    }
    // RFC 3533 §6: an Ogg page begins with the capture pattern "OggS". The first page of a
    // logical bitstream (BOS) carries the codec identification header as its first packet.
    if b.starts_with(b"OggS") {
        return ogg_bos(b);
    }
    // ISO/IEC 11172-3 §2.4.3.1 (and 13818-3 §2.4.3.1): an MPEG audio frame header starts
    // with a 12-bit syncword of all ones — in practice tested as 11 bits, since MPEG-2.5
    // (the de-facto extension) encodes its version in the 12th. Two reserved encodings rule
    // out a false positive on ADTS AAC, whose syncword is the same 12 bits (ISO/IEC
    // 14496-3 §1.A.2.2.1) but whose `layer` field is always 00 — reserved here.
    if b.len() >= 2 && b[0] == 0xFF && (b[1] & 0xE0) == 0xE0 {
        let version = (b[1] >> 3) & 0x03; // 01 = reserved (§2.4.2.3, "ID")
        let layer = (b[1] >> 1) & 0x03; //   00 = reserved (§2.4.2.3, "layer")
        if version != 0b01 && layer != 0b00 {
            return Format::Mp3;
        }
    }
    Format::Unknown
}

/// Classify an Ogg physical bitstream by the codec identification header in its first page.
///
/// RFC 3533 §6 page header: `"OggS"`, version (1), header_type (1), granule position (8),
/// bitstream serial (4), page sequence (4), CRC (4), page_segments (1) — 27 bytes — then
/// `page_segments` lacing bytes, then the payload. The BOS page carries the codec's
/// identification header at the start of that payload (§6, "the first page of a logical
/// bitstream").
fn ogg_bos(page: &[u8]) -> Format {
    let Some(&nsegs) = page.get(26) else { return Format::Unknown };
    let payload = 27 + nsegs as usize;
    let Some(id) = page.get(payload..) else { return Format::Unknown };
    // RFC 7845 §5.1: the Opus identification header begins with the magic "OpusHead".
    if id.starts_with(b"OpusHead") {
        return Format::OggOpus;
    }
    // Vorbis I specification §4.2.1: every Vorbis header packet is a packet-type byte
    // followed by "vorbis"; the identification header is type 1.
    if id.starts_with(b"\x01vorbis") {
        return Format::OggVorbis;
    }
    // "Ogg Mapping for FLAC" §3 (mapping version 1): the first packet is 0x7F, "FLAC",
    // the mapping version, the header-packet count, then a native `fLaC` stream.
    if id.starts_with(b"\x7fFLAC") {
        return Format::OggFlac;
    }
    // Speex manual §7.3: the header packet's `speex_string` "must contain the 'Speex   '
    // (with 3 trailing spaces), which identifies the bit-stream". Unlike the three above
    // there is no packet-type octet — the eight-character magic, trailing spaces included, is
    // the whole signal (`ogg/spec/SPEEX.md`).
    if id.starts_with(pf_ogg::ident::SPEEX_MAGIC) {
        return Format::OggSpeex;
    }
    Format::Unknown
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;

    fn ogg_page(id_header: &[u8]) -> Vec<u8> {
        let mut p = b"OggS".to_vec();
        p.extend_from_slice(&[0, 2]); // version 0, header_type BOS
        p.extend_from_slice(&[0u8; 8]); // granule
        p.extend_from_slice(&[0u8; 8]); // serial + seq
        p.extend_from_slice(&[0u8; 4]); // crc
        p.push(1); // one lacing value
        p.push(id_header.len() as u8);
        p.extend_from_slice(id_header);
        p
    }

    #[test]
    fn magics() {
        assert_eq!(sniff(b"fLaC\x00\x00\x00\x22").format, Format::Flac);
        let mut wav = b"RIFF".to_vec();
        wav.extend_from_slice(&[0, 0, 0, 0]);
        wav.extend_from_slice(b"WAVEfmt ");
        assert_eq!(sniff(&wav).format, Format::Wav);
        assert_eq!(sniff(b"\x00\x00\x00\x20ftypM4A \x00\x00").format, Format::Mp4);
        assert_eq!(sniff(&[0x1A, 0x45, 0xDF, 0xA3, 1, 2, 3, 4]).format, Format::Mkv);
        assert_eq!(sniff(&ogg_page(b"OpusHead...")).format, Format::OggOpus);
        assert_eq!(sniff(&ogg_page(b"\x01vorbis..")).format, Format::OggVorbis);
        assert_eq!(sniff(&ogg_page(b"\x7fFLAC\x01\x00")).format, Format::OggFlac);
        assert_eq!(sniff(&[0xFF, 0xFB, 0x90, 0x00]).format, Format::Mp3);
    }

    #[test]
    fn magics_of_the_apev2_family_and_the_iff_forms() {
        let mut aiff = b"FORM".to_vec();
        aiff.extend_from_slice(&[0, 0, 1, 0]);
        aiff.extend_from_slice(b"AIFFCOMM");
        assert_eq!(sniff(&aiff).format, Format::Aiff);
        aiff[8..12].copy_from_slice(b"AIFC");
        assert_eq!(sniff(&aiff).format, Format::Aiff);
        // A `FORM` that is not audio must not be claimed.
        aiff[8..12].copy_from_slice(b"8SVX");
        assert_eq!(sniff(&aiff).format, Format::Unknown);

        assert_eq!(sniff(b"wvpk\x20\x01\x00\x00\x02\x04").format, Format::WavPack);
        assert_eq!(sniff(b"MAC \x96\x0f\x00\x00").format, Format::Ape);
        assert_eq!(sniff(b"MACF\x96\x0f\x00\x00").format, Format::Ape);
        assert_eq!(sniff(b"MPCK SH").format, Format::Musepack);
        assert_eq!(sniff(b"MP+\x07\x10\x00\x00\x00").format, Format::Musepack);
        assert_eq!(sniff(b"MP+\x17\x10\x00\x00\x00").format, Format::Musepack, "PNS nibble set");
        // The low nibble is the stream version: 8 is not SV7.
        assert_eq!(sniff(b"MP+\x08\x10\x00\x00\x00").format, Format::Unknown);
        assert_eq!(sniff(&ogg_page(b"Speex   1.2rc1")).format, Format::OggSpeex);
        // Two trailing spaces is not the magic.
        assert_eq!(sniff(&ogg_page(b"Speex  x1.2rc1")).format, Format::Unknown);
    }

    /// An ID3v2 tag in front of an APEv2-family stream is legal and occurs in the wild; the
    /// re-sniff must see through it rather than falling back to "probably MP3".
    #[test]
    fn an_id3_prefixed_apev2_family_file_is_not_mp3() {
        for (magic, want) in [
            (&b"MAC \x96\x0f\x00\x00"[..], Format::Ape),
            (&b"wvpk\x20\x01\x00\x00"[..], Format::WavPack),
            (&b"MPCK SH"[..], Format::Musepack),
            (&b"MP+\x07\x10\x00\x00"[..], Format::Musepack),
        ] {
            // "ID3" v2.4, no flags, synchsafe size 10 → the stream begins at 20.
            let mut f = b"ID3\x04\x00\x00\x00\x00\x00\x0a".to_vec();
            f.extend_from_slice(&[0u8; 10]);
            f.extend_from_slice(magic);
            let s = sniff(&f);
            assert_eq!(s.format, want, "{magic:?}");
            assert_eq!(s.body, 20, "the parser is pointed at the real header");
        }
    }

    #[test]
    fn adts_aac_is_not_mp3() {
        // 0xFFF1: sync + MPEG-4 + layer 00 (reserved for MPEG audio) — ADTS, not MP3.
        assert_eq!(sniff(&[0xFF, 0xF1, 0x50, 0x80]).format, Format::Unknown);
    }

    #[test]
    fn id3_prefix_is_skipped() {
        // "ID3" v2.4, no flags, synchsafe size 0x0A = 10 → body at 20.
        let mut f = b"ID3\x04\x00\x00\x00\x00\x00\x0a".to_vec();
        f.extend_from_slice(&[0u8; 10]);
        f.extend_from_slice(b"fLaC");
        let s = sniff(&f);
        assert_eq!(s.format, Format::Flac);
        assert_eq!(s.body, 20);

        // Same tag in front of nothing recognisable → MP3 (the overwhelmingly likely case).
        let mut m = b"ID3\x04\x00\x00\x00\x00\x00\x0a".to_vec();
        m.extend_from_slice(&[0u8; 10]);
        m.extend_from_slice(&[0xFF, 0xFB, 0x90, 0x00]);
        assert_eq!(sniff(&m).format, Format::Mp3);
    }

    #[test]
    fn id3_footer_flag_adds_ten() {
        let f = b"ID3\x04\x00\x10\x00\x00\x00\x0a";
        assert_eq!(id3v2_len(f), Some(30));
    }

    #[test]
    fn malformed_heads_never_panic() {
        for len in 0..40usize {
            let mut v = vec![0xFFu8; len];
            assert!(sniff(&v).format.name().is_ascii());
            v.iter_mut().enumerate().for_each(|(i, b)| *b = (i * 7) as u8);
            let _ = sniff(&v);
        }
        assert_eq!(sniff(b"OggS").format, Format::Unknown);
        assert_eq!(sniff(b"ID3").format, Format::Unknown);
    }
}
