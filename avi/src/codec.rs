//! Stream-format → src-pad **family** mapping and per-track offer menus, mirroring
//! `pf-mkv`'s `codec::family_for` / `offers_for`. An AVI stream's codec is named two ways
//! depending on the medium (AVI RIFF File Reference):
//! - **video** by the `strh.fccHandler` / `strf.biCompression` **fourcc** (`XVID`, `DX50`,
//!   `H264`, …);
//! - **audio** by the `strf` WAVEFORMATEX **`wFormatTag`** (a registered 16-bit id — see the
//!   Microsoft `mmreg.h` "WAVE_FORMAT_*" registry and the AC-3/DTS/MP3 assignments).
//!
//! The families are `&'static str` and must match the sink offers of the profluens
//! **decoders** built by the other agents (this crate's brief pins the vocabulary):
//! `mpeg4/asp`, `h264/annexb`, `ac3`, `mp3`, `audio/raw`. A stream we cannot classify maps
//! to `bytes` (the escape a generic byte sink still taps; a warning is logged and downstream
//! decode is skipped). Per-track menus are what make negotiation-driven autoplug *select* the
//! right decoder (the same lesson `pf-mkv` documents: one shared all-families menu let the
//! wrong decoder link, then drop every packet at runtime).

use profluens_core::format::OfferDesc;

use crate::riff::{Stream, StreamKind};

/// The `mpeg4/asp` family: MPEG-4 Part 2 Advanced Simple Profile (XviD/DivX). The fourccs
/// below all name Part-2 ASP bitstreams; the case-insensitive match covers the `xvid`/`XVID`
/// spelling variants real muxers write (ISO/IEC 14496-2; the fourcc registry at
/// `fourcc.org`).
pub const FAMILY_MPEG4_ASP: &str = "mpeg4/asp";
/// The `h264/annexb` family: H.264 in Annex B framing (rare in AVI, best-effort — the
/// `H264`/`avc1` fourccs). AVI stores H.264 without a length-prefix/config record, so it is
/// usually already Annex B; we forward verbatim.
pub const FAMILY_H264_ANNEXB: &str = "h264/annexb";
/// The `ac3` family: Dolby AC-3 (and DTS, which shares this family here) — WAVEFORMATEX
/// `wFormatTag` 0x2000 (AC-3) / 0x2001 (DTS). This is the Nord file's audio.
pub const FAMILY_AC3: &str = "ac3";
/// The `mp3` family: MPEG-1/2 Audio Layer III — `wFormatTag` 0x0055 (WAVE_FORMAT_MPEGLAYER3).
pub const FAMILY_MP3: &str = "mp3";
/// The `audio/raw` family: linear PCM — `wFormatTag` 0x0001 (WAVE_FORMAT_PCM). Announced with
/// rate/channels/sample so a raw sink links directly.
pub const FAMILY_AUDIO_RAW: &str = "audio/raw";
/// The `bytes` escape for an unclassified stream.
pub const FAMILY_BYTES: &str = "bytes";

/// The announce family for a discovered [`Stream`] (see the module docs). Video keys on the
/// fourcc (case-insensitive), audio on the WAVEFORMATEX tag.
pub fn family_for(stream: &Stream) -> &'static str {
    match stream.kind {
        StreamKind::Video => video_family(stream.handler, &stream.strf),
        StreamKind::Audio => audio_family(stream.format_tag),
        StreamKind::Other => FAMILY_BYTES,
    }
}

/// Video family from the `strh.fccHandler` (and, as a fallback, the `strf.biCompression`
/// fourcc at BITMAPINFOHEADER offset 16). Case-insensitive (ISO/IEC 14496-2; `fourcc.org`).
fn video_family(handler: crate::riff::FourCc, strf: &[u8]) -> &'static str {
    // Try the handler, then the biCompression fourcc — some muxers leave fccHandler zero and
    // only set biCompression (AVI RIFF Reference notes both name the codec).
    let bicompression: crate::riff::FourCc =
        strf.get(16..20).map(|s| [s[0], s[1], s[2], s[3]]).unwrap_or([0; 4]);
    for cc in [handler, bicompression] {
        match &fourcc_upper(cc) {
            // MPEG-4 Part 2 ASP spellings (XviD, DivX 5, generic MP4V/FMP4/DIVX).
            b"XVID" | b"DX50" | b"DIVX" | b"DX40" | b"FMP4" | b"MP4V" | b"MP43" => {
                return FAMILY_MPEG4_ASP
            }
            b"H264" | b"AVC1" | b"X264" | b"DAVC" => return FAMILY_H264_ANNEXB,
            _ => {}
        }
    }
    FAMILY_BYTES
}

/// Audio family from the WAVEFORMATEX `wFormatTag` (Microsoft `mmreg.h` WAVE_FORMAT_* ids).
fn audio_family(format_tag: u16) -> &'static str {
    match format_tag {
        0x2000 | 0x2001 => FAMILY_AC3, // WAVE_FORMAT_DOLBY_AC3 (0x2000) / DTS (0x2001)
        0x0055 => FAMILY_MP3,          // WAVE_FORMAT_MPEGLAYER3
        0x0001 => FAMILY_AUDIO_RAW,    // WAVE_FORMAT_PCM
        _ => FAMILY_BYTES,
    }
}

/// Upper-case a fourcc's ASCII letters so `xvid` and `XVID` match (non-letters unchanged).
fn fourcc_upper(cc: crate::riff::FourCc) -> crate::riff::FourCc {
    [
        cc[0].to_ascii_uppercase(),
        cc[1].to_ascii_uppercase(),
        cc[2].to_ascii_uppercase(),
        cc[3].to_ascii_uppercase(),
    ]
}

/// Whether a family names a stream a downstream decoder can handle (i.e. not the `bytes`
/// escape). A `bytes` track links only to a generic byte sink; the element warns on it.
pub fn is_decodable(family: &str) -> bool {
    family != FAMILY_BYTES
}

/// Whether a family is a video family — decides whether a `width`/`height` announcement is
/// meaningful and whether post-seek keyframe gating applies.
pub fn is_video_family(family: &str) -> bool {
    matches!(family, "mpeg4/asp" | "h264/annexb")
}

/// The per-track src-pad offer menu: exactly the stream's [`family_for`] family plus the
/// `bytes` escape, so link-time negotiation *selects* the matching decoder while a generic
/// byte peer can still tap the track. `any` offers — the concrete params ride the runtime
/// announcement (rate/channels/width/height), whose field names the consumer's offers intern
/// (the `pf-mkv` per-track-menu lesson).
pub fn offers_for(stream: &Stream) -> &'static [OfferDesc] {
    static MPEG4: [OfferDesc; 2] = [OfferDesc::any("mpeg4/asp"), OfferDesc::any("bytes")];
    static H264: [OfferDesc; 2] = [OfferDesc::any("h264/annexb"), OfferDesc::any("bytes")];
    static AC3: [OfferDesc; 2] = [OfferDesc::any("ac3"), OfferDesc::any("bytes")];
    static MP3: [OfferDesc; 2] = [OfferDesc::any("mp3"), OfferDesc::any("bytes")];
    static RAW: [OfferDesc; 2] = [OfferDesc::any("audio/raw"), OfferDesc::any("bytes")];
    static BYTES: [OfferDesc; 1] = [OfferDesc::any("bytes")];
    match family_for(stream) {
        FAMILY_MPEG4_ASP => &MPEG4,
        FAMILY_H264_ANNEXB => &H264,
        FAMILY_AC3 => &AC3,
        FAMILY_MP3 => &MP3,
        FAMILY_AUDIO_RAW => &RAW,
        _ => &BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::riff::{FourCc, Stream, StreamKind};

    fn video(handler: &[u8; 4], bicompression: Option<&[u8; 4]>) -> Stream {
        let mut strf = vec![0u8; 40];
        if let Some(cc) = bicompression {
            strf[16..20].copy_from_slice(cc);
        }
        Stream {
            index: 0,
            kind: StreamKind::Video,
            handler: *handler,
            scale: 1,
            rate: 24,
            sample_size: 0,
            length: 0,
            width: 1280,
            height: 544,
            strf,
            format_tag: 0,
            channels: 0,
            samples_per_sec: 0,
            bits_per_sample: 0,
        }
    }

    fn audio(tag: u16) -> Stream {
        Stream {
            index: 1,
            kind: StreamKind::Audio,
            handler: [0; 4],
            scale: 1,
            rate: 48_000,
            sample_size: 0,
            length: 0,
            width: 0,
            height: 0,
            strf: Vec::new(),
            format_tag: tag,
            channels: 6,
            samples_per_sec: 48_000,
            bits_per_sample: 0,
        }
    }

    #[test]
    fn xvid_variants_map_to_mpeg4_asp() {
        for cc in [b"XVID", b"xvid", b"DX50", b"DIVX", b"FMP4", b"MP4V"] {
            let cc: &[u8; 4] = cc;
            assert_eq!(family_for(&video(cc, None)), "mpeg4/asp", "{cc:?}");
        }
    }

    #[test]
    fn bicompression_fallback_classifies_video() {
        // fccHandler zeroed, codec only in biCompression.
        let s = video(&[0, 0, 0, 0], Some(b"XVID"));
        assert_eq!(family_for(&s), "mpeg4/asp");
    }

    #[test]
    fn h264_fourccs_map_to_annexb() {
        assert_eq!(family_for(&video(b"H264", None)), "h264/annexb");
        assert_eq!(family_for(&video(b"avc1", None)), "h264/annexb");
    }

    #[test]
    fn unknown_video_fourcc_is_bytes() {
        assert_eq!(family_for(&video(b"MJPG", None)), "bytes");
    }

    #[test]
    fn audio_tags_map_to_families() {
        assert_eq!(family_for(&audio(0x2000)), "ac3", "AC-3");
        assert_eq!(family_for(&audio(0x2001)), "ac3", "DTS shares the ac3 family");
        assert_eq!(family_for(&audio(0x0055)), "mp3");
        assert_eq!(family_for(&audio(0x0001)), "audio/raw");
        assert_eq!(family_for(&audio(0x00FF)), "bytes", "unknown tag → bytes");
    }

    #[test]
    fn offer_menus_carry_family_plus_bytes() {
        let menu = offers_for(&audio(0x2000));
        assert_eq!(menu.len(), 2);
        assert_eq!(menu[0].family, "ac3");
        assert_eq!(menu[1].family, "bytes");
        let bmenu = offers_for(&audio(0xBEEF));
        assert_eq!(bmenu.len(), 1);
        assert_eq!(bmenu[0].family, "bytes");
    }

    #[test]
    fn decodability_and_video_predicate() {
        assert!(is_decodable("mpeg4/asp"));
        assert!(!is_decodable("bytes"));
        assert!(is_video_family("mpeg4/asp"));
        assert!(is_video_family("h264/annexb"));
        assert!(!is_video_family("ac3"));
    }

    // Silence unused-import style warnings for the type alias in doc tests.
    #[allow(dead_code)]
    fn _touch(_: FourCc) {}
}
