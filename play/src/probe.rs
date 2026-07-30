//! Typefind: name the container/elementary format from a small prefix read (spec:
//! Milestone applications §5 — a player is handed a file and must decide how to build
//! the graph before any pipeline exists). This is pure controller logic — a magic-byte
//! classifier over the file's first few bytes — the honest v1 of GStreamer's `typefind`.
//!
//! Every discriminator cites the spec that fixes the magic at the point of use (repo
//! standing rule: a non-trivial constant names its authority; see the `pf-mkv`
//! `ebml`/`codec` modules and the `pf-mp3` `id3v2_len` for the tone). The order below is
//! deliberate: the containers whose magic sits at a fixed offset (EBML, ISO-BMFF, Ogg,
//! FLAC, RIFF) are unambiguous and tried first; the elementary MPEG-audio classifiers
//! (ID3 tag / raw sync, ADTS) are self-syncing byte streams and come last so a container
//! that merely happens to embed an MPEG frame is never misfiled.

/// The formats [`probe`] can name — the vocabulary the rest of the crate switches on.
/// Container vs elementary is a build-graph distinction (a container gets a demuxer and
/// dynamic pads; an elementary stream is `filesrc → parser/decoder` with no demux),
/// captured by [`Kind::is_elementary`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// Matroska / WebM (EBML). WebM is a Matroska subset, so one demuxer covers both.
    Mkv,
    /// ISO Base Media File Format (`.mp4`, `.m4a`, `.mov`).
    Mp4,
    /// Ogg (the container). The first logical bitstream's codec is discovered later, by
    /// running the demuxer (v1: FLAC-in-Ogg decodes; Vorbis/Opus/Theora drop cleanly).
    Ogg,
    /// AVI (RIFF). The classic `.avi` container — XviD/DivX (MPEG-4 Part 2) video + AC-3/MP3
    /// audio in the wild. One demuxer (`pf-avi`) covers it; tracks discovered at preroll.
    Avi,
    /// A raw native FLAC stream (`.flac`), no container.
    Flac,
    /// A raw MPEG-1/2/2.5 audio elementary stream (`.mp3`), optionally ID3-tagged.
    Mp3,
    /// A raw ADTS-framed AAC elementary stream (`.aac`/`.adts`).
    AdtsAac,
    /// A RIFF/WAVE PCM file (`.wav`).
    Wav,
}

impl Kind {
    /// Whether this is an elementary (container-less) stream: `filesrc` feeds the
    /// parser/decoder directly, with no demux stage and no dynamic pads.
    pub fn is_elementary(self) -> bool {
        matches!(self, Kind::Flac | Kind::Mp3 | Kind::AdtsAac | Kind::Wav)
    }

    /// A short human label for the track/summary printout.
    pub fn label(self) -> &'static str {
        match self {
            Kind::Mkv => "Matroska/WebM",
            Kind::Mp4 => "ISO-BMFF (MP4)",
            Kind::Ogg => "Ogg",
            Kind::Avi => "AVI (RIFF)",
            Kind::Flac => "FLAC",
            Kind::Mp3 => "MP3",
            Kind::AdtsAac => "ADTS AAC",
            Kind::Wav => "WAV (RIFF/WAVE)",
        }
    }
}

/// How many prefix bytes [`probe`] needs to decide. Every fixed-offset magic lives in the
/// first 16 bytes (the RIFF probe reads `WAVE` at offset 8, the longest reach); the ID3
/// tag and the ADTS/MPEG sync words are all inside the first 4. 64 is generous slack for
/// a leading BOM or whitespace some junk producers prepend before an MPEG stream.
pub const PREFIX_LEN: usize = 64;

/// The list of formats [`probe`] can name, for the "unknown → here is what IS supported"
/// error. One source of truth, so the error can never drift from the match arms.
const SUPPORTED: &str =
    "Matroska/WebM (.mkv/.webm), MP4/ISO-BMFF (.mp4/.m4a/.mov), Ogg (.ogg/.oga), \
     AVI (.avi), FLAC (.flac), MP3 (.mp3), ADTS AAC (.aac/.adts), WAV (.wav)";

/// Classify a file from its leading bytes. `Err` names what the byte pattern was and lists
/// the supported formats, so the CLI can print one clear line and exit nonzero.
///
/// `prefix` should hold at least the first [`PREFIX_LEN`] bytes of the file (fewer is fine
/// — a short file simply can't match a magic that reaches past its end).
pub fn probe(prefix: &[u8]) -> Result<Kind, String> {
    // --- Fixed-offset container/elementary magics (unambiguous) ---

    // EBML header id `1A 45 DF A3` at offset 0 (Matroska: IETF RFC 9559 §4; EBML:
    // RFC 8794 §5). Matches both `.mkv` and `.webm` (WebM is an EBML/Matroska profile).
    if prefix.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Ok(Kind::Mkv);
    }

    // ISO-BMFF: a `ftyp` box at the very start, i.e. the four-CC `ftyp` at offset 4
    // (ISO/IEC 14496-12 §4.3 — the File Type Box is required and, in a streamable file,
    // first). Bytes 0..4 are its (big-endian) box size, not checked here.
    if prefix.len() >= 8 && &prefix[4..8] == b"ftyp" {
        return Ok(Kind::Mp4);
    }

    // Ogg capture pattern `OggS` at offset 0 (RFC 3533 §6, page header field 1).
    if prefix.starts_with(b"OggS") {
        return Ok(Kind::Ogg);
    }

    // Native FLAC stream marker `fLaC` at offset 0 (RFC 9639 §8.1 / the FLAC format spec
    // §"STREAM": the 32-bit magic before the first metadata block).
    if prefix.starts_with(b"fLaC") {
        return Ok(Kind::Flac);
    }

    // RIFF containers share the `RIFF` magic at 0 and a 4-byte little-endian file size,
    // then a 4-CC *form type* at offset 8 that disambiguates them (Microsoft/IBM Multimedia
    // Programming Interface, RIFF §2). `AVI ` (trailing space) → AVI; `WAVE` → WAV.
    if prefix.len() >= 12 && &prefix[0..4] == b"RIFF" {
        // AVI: form type `AVI ` (Microsoft AVI RIFF File Reference — the `RIFF('AVI ' …)`
        // top-level form). The trailing space is part of the 4-CC.
        if &prefix[8..12] == b"AVI " {
            return Ok(Kind::Avi);
        }
        // WAVE: the form `wavparse` strips.
        if &prefix[8..12] == b"WAVE" {
            return Ok(Kind::Wav);
        }
    }

    // --- Elementary MPEG-audio streams (self-syncing byte magics) ---

    // ADTS AAC: a 12-bit syncword `1111 1111 1111` then the MPEG version + layer bits,
    // giving the well-known first bytes `FF F1` (MPEG-4, no CRC / with CRC → the low bit
    // of byte 1 varies) or `FF F9` (MPEG-2) — ISO/IEC 14496-3 §1.A.2.2.1, `syncword` +
    // `ID` + `layer` (layer is always `00`). Checked before the looser MP3 sync so an
    // ADTS stream is never misfiled as MP3 (both start `FF Fx`; ADTS pins layer `00`,
    // MP3 pins a non-`00` layer — see below).
    if prefix.len() >= 2 && prefix[0] == 0xFF && matches!(prefix[1], 0xF1 | 0xF9) {
        return Ok(Kind::AdtsAac);
    }

    // MP3, ID3-tagged form: a leading `ID3` tag (id3.org ID3v2.4 §3.1 — `"ID3"` then two
    // version bytes). The audio follows the tag; the byte reader in `pf-mp3` skips the
    // tag, so naming it MP3 here is enough.
    if prefix.starts_with(b"ID3") {
        return Ok(Kind::Mp3);
    }

    // MP3, bare form: an 11-bit frame sync `1111 1111 111` (`FF Ex`) whose layer field is
    // *not* `00` — Layer I/II/III (ISO/IEC 11172-3 / 13818-3 §2.4.2.3; layer bits are
    // byte 1 bits 2..1, `00` == "reserved", which is ADTS's marker). This rejects the
    // ADTS case (handled above) while accepting `FF FB` (MPEG-1 Layer III), `FF F3`,
    // `FF F2`, `FF E3`, etc.
    if prefix.len() >= 2 && prefix[0] == 0xFF && (prefix[1] & 0xE0) == 0xE0 {
        let layer = (prefix[1] >> 1) & 0x03;
        if layer != 0b00 {
            return Ok(Kind::Mp3);
        }
    }

    Err(format!(
        "unrecognized file: leading bytes {:02X?} match no supported container or codec. \
         Supported: {SUPPORTED}",
        &prefix[..prefix.len().min(8)],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a prefix from a magic plus filler, so a test vector is exactly the bytes a
    /// real file would start with (and long enough for the offset-8/12 probes).
    fn prefix(head: &[u8]) -> Vec<u8> {
        let mut v = head.to_vec();
        v.resize(PREFIX_LEN, 0);
        v
    }

    #[test]
    fn table_of_magics() {
        // (name, leading bytes, expected). Each row's bytes are the real spec magic at
        // its real offset; a regression that widens/narrows a discriminator trips here.
        let cases: &[(&str, &[u8], Kind)] = &[
            ("mkv EBML", &[0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00], Kind::Mkv),
            ("mp4 ftyp isom", b"\x00\x00\x00\x20ftypisom", Kind::Mp4),
            ("mp4 ftyp mp42", b"\x00\x00\x00\x18ftypmp42", Kind::Mp4),
            ("mov ftyp qt", b"\x00\x00\x00\x14ftypqt  ", Kind::Mp4),
            ("ogg", b"OggS\x00\x02", Kind::Ogg),
            ("flac", b"fLaC\x00\x00\x00\x22", Kind::Flac),
            ("wav", b"RIFF\x24\x08\x00\x00WAVEfmt ", Kind::Wav),
            // ADTS: FF F1 (MPEG-4, no CRC) and FF F9 (MPEG-2).
            ("adts mpeg4", &[0xFF, 0xF1, 0x50, 0x80], Kind::AdtsAac),
            ("adts mpeg2", &[0xFF, 0xF9, 0x4C, 0x80], Kind::AdtsAac),
            // MP3: ID3 tag, and bare MPEG-1 Layer III (FF FB) / Layer II (FF FC-ish).
            ("mp3 id3", b"ID3\x04\x00\x00", Kind::Mp3),
            ("mp3 bare L3", &[0xFF, 0xFB, 0x90, 0x00], Kind::Mp3),
            ("mp3 bare mpeg2 L3", &[0xFF, 0xF3, 0x82, 0xC4], Kind::Mp3),
        ];
        for (name, bytes, want) in cases {
            assert_eq!(probe(&prefix(bytes)), Ok(*want), "case {name}");
        }
    }

    #[test]
    fn adts_and_mp3_are_disambiguated_by_layer() {
        // FF F1 / FF F9 have layer bits 00 → ADTS; FF FB / FF F3 have a non-00 layer → MP3.
        // The two families share the top of the syncword, so the layer bit is the pivot.
        assert_eq!(probe(&[0xFF, 0xF1]), Ok(Kind::AdtsAac));
        assert_eq!(probe(&[0xFF, 0xFB]), Ok(Kind::Mp3));
        // A syncword with the *reserved* layer `00` that is not ADTS's F1/F9 is claimed as
        // neither (leaves the door to a clean "unknown"). `FF E1` = MPEG-2.5, layer 00 —
        // a structurally invalid MPEG frame, so we do not misfile it as MP3.
        assert!(probe(&[0xFF, 0xE1]).is_err(), "reserved layer 00, not ADTS's F1/F9");
    }

    #[test]
    fn riff_avi_is_avi_not_wav() {
        // A RIFF whose form is `AVI ` is the AVI container, never misfiled as WAV.
        let avi = prefix(b"RIFF\x00\x00\x00\x00AVI LIST");
        assert_eq!(probe(&avi), Ok(Kind::Avi), "RIFF/`AVI ` is the AVI container");
        // A RIFF with an unknown form (neither WAVE nor AVI) is still unrecognized.
        let other = prefix(b"RIFF\x00\x00\x00\x00XYZ ");
        assert!(probe(&other).is_err(), "an unknown RIFF form is not a supported format");
    }

    #[test]
    fn unknown_lists_supported_formats() {
        let err = probe(b"\x89PNG\r\n\x1a\n").unwrap_err();
        assert!(err.contains("Matroska"), "error should name supported formats: {err}");
        assert!(err.contains("MP4"), "error should name supported formats: {err}");
    }

    #[test]
    fn short_prefixes_do_not_panic() {
        // A file shorter than a magic's reach simply doesn't match it — never an index panic.
        for n in 0..12 {
            let _ = probe(&[0xFFu8; 12][..n]);
            let _ = probe(&b"RIFF"[..n.min(4)]);
        }
    }

    #[test]
    fn is_elementary_partitions_the_kinds() {
        for k in [Kind::Flac, Kind::Mp3, Kind::AdtsAac, Kind::Wav] {
            assert!(k.is_elementary(), "{k:?} is elementary");
        }
        for k in [Kind::Mkv, Kind::Mp4, Kind::Ogg] {
            assert!(!k.is_elementary(), "{k:?} is a container");
        }
    }
}
