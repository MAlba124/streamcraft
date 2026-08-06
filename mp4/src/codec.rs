//! Sample-entry parsing (`stsd`, §8.5.2) and the box-type → announce-family map (spec:
//! ISO/IEC 14496-12 §8.5.2; RFC 6381 codecs strings at `mp4/spec/rfc6381.txt`).
//!
//! The `stsd` SampleDescriptionBox holds one or more **sample entries** — a
//! `VisualSampleEntry` (§12.1.3) for video or an `AudioSampleEntry` (§12.2.3) for audio —
//! each a box whose type is the codec four-CC (`avc1`, `hvc1`, `vp09`, `av01`, `Opus`,
//! `mp4a`, …) and whose payload nests a codec-configuration box (`avcC`, `hvcC`, `vpcC`,
//! `av1C`, `dOps`, `esds`).
//!
//! This module extracts exactly what the demuxer announces and reframes with:
//! - the announce **family** for the profluens decoders ([`family_for`]);
//! - the visual **dimensions** / audio **rate/channels** from the sample entry;
//! - the codec **configuration record** (`avcC`/`hvcC`) that seeds the Annex B reframer,
//!   reusing `pf-mkv`'s [`nal_head_from_config`](pf_mkv::nal_head_from_config) parser (the
//!   identical ISO/IEC 14496-15 record rides both an MKV `CodecPrivate` and an MP4
//!   `avc1`/`hvc1` sample entry — one parser, no copy; see `mp4/NOTES.md`).
//!
//! ## Untrusted input (spec: "a crash on bad input is a P0")
//! Every field is bounds-checked; a malformed sample entry yields a [`BoxError`], and a
//! malformed config record degrades to an empty Annex B head (the pad still links — the
//! decoder resyncs at an in-band parameter set), never a panic.

// COLD: stsd/sample-entry/esds parsing — runs once per stream at preroll, building
// codec-head/config-record buffers that outlive process(); no per-sample hot path here.
#![allow(clippy::disallowed_methods)]

use crate::boxes::{self, BoxError, BoxHeader};

pub use pf_mkv::{nal_head_from_config, Reframer};

/// A parsed sample entry — the codec four-CC and the format params + reframing info the
/// demuxer needs to announce the pad and slice frames. Built from one `stsd` child box.
#[derive(Clone, Debug)]
pub struct SampleEntry {
    /// The sample-entry box type (`avc1`, `vp09`, `mp4a`, …) as raw octets.
    pub kind: boxes::FourCc,
    /// The announce family for the profluens decoders ([`family_for`]).
    pub family: &'static str,
    /// Video: coded width in pixels (0 for audio / unknown).
    pub width: u32,
    /// Video: coded height in pixels (0 for audio / unknown).
    pub height: u32,
    /// Audio: sample rate in Hz (0 for video / unknown).
    pub sample_rate: u32,
    /// Audio: channel count (0 for video / unknown).
    pub channels: u32,
    /// The Annex B parameter-set head to emit before the first frame (empty for non-NAL
    /// codecs, for `avc3`/`hev1` in-band parameter sets, or a malformed config record).
    pub codec_head: Vec<u8>,
    /// The **raw** codec-configuration record, verbatim — what a remux hands to Matroska
    /// as `CodecPrivate` (RFC 9559 §12). For NAL codecs it is the `avcC`/`hvcC` box body
    /// (ISO/IEC 14496-15: `V_MPEG4/ISO/AVC` CodecPrivate *is* the
    /// AVCDecoderConfigurationRecord), kept even when [`nal_head_from_config`] rejects it
    /// (a remux is byte-faithful; only the decode path needs to parse it). For `mp4a` AAC
    /// it is the `esds`-carried **AudioSpecificConfig** (ISO/IEC 14496-3 §1.6.2.1 —
    /// Matroska A_AAC CodecPrivate). Empty for other codecs and in-band-only `avc3`/`hev1`.
    pub config_record: Vec<u8>,
    /// How each sample's payload is reframed downstream: NAL length-prefix → Annex B for
    /// `avc*`/`hvc*`/`hev*`, passthrough for everything else.
    pub reframer: Reframer,
}

/// The demuxer announce family for a sample-entry box type (§8.5.2; RFC 6381). The families
/// match the sink offers of the profluens decoders so a dynamic pad linking
/// `mp4demux.src_track1 ! h264dec.sink` (etc.) negotiates:
/// - `avc1`/`avc3` → `h264/annexb`, `hvc1`/`hev1` → `h265/annexb`;
/// - `vp09` → `vp9`, `av01` → `av1`;
/// - `Opus` → `opus`; `mp4a` → `bytes` *here*, promoted to `aac` by
///   [`parse_audio_entry`] once an `esds`-carried AudioSpecificConfig is in hand (the
///   family is only meaningful with the ASC — a remuxing consumer needs it as
///   `CodecPrivate`, RFC 9559 §12 A_AAC — so an ASC-less `mp4a` stays a raw `bytes` pad);
/// - anything else → `bytes` (the unknown-codec fallback).
///
/// The families are `&'static str` so they can seed a pad's static offer menu (a pad may
/// only announce a family it already offered — the vocabulary is interned from the offers).
pub fn family_for(kind: boxes::FourCc) -> &'static str {
    match &kind {
        b"avc1" | b"avc3" => "h264/annexb",
        b"hvc1" | b"hev1" => "h265/annexb",
        b"vp09" => "vp9",
        b"av01" => "av1",
        b"Opus" => "opus",
        // AAC in `mp4a`: `bytes` until the esds yields an AudioSpecificConfig (see above).
        b"mp4a" => "bytes",
        _ => "bytes",
    }
}

/// Whether a family names a video codec (so a `width`/`height` announcement is meaningful).
pub fn is_video_family(family: &str) -> bool {
    matches!(family, "vp9" | "av1" | "h264/annexb" | "h265/annexb")
}

/// Whether a sample-entry type carries **length-prefixed NAL units** that must be reframed
/// to Annex B (§8.5.2; ISO/IEC 14496-15). `avc3`/`hev1` still length-prefix their NALs — the
/// difference from `avc1`/`hvc1` is only that their parameter sets are in-band, so we reframe
/// the length prefixes but emit no `avcC`/`hvcC` head.
fn is_nal_entry(kind: &boxes::FourCc) -> bool {
    matches!(kind, b"avc1" | b"avc3" | b"hvc1" | b"hev1")
}

/// Parse the first sample entry of an `stsd` payload (§8.5.2). A conforming media track has
/// exactly one sample entry; if several are present (rare, e.g. mid-stream format changes)
/// only the first is used — the resolver's `sample_description_index` from `stsc` is
/// validated against the count but a track that switches sample entry mid-stream is out of
/// v1 scope (documented in `mp4/NOTES.md`).
///
/// Layout: FullBox head, `entry_count` u32, then the entries. We read the first entry's box
/// header and dispatch on its type.
pub fn parse_stsd(body: &[u8]) -> Result<SampleEntry, BoxError> {
    // FullBox head (4) + entry_count (4) = 8 octets before the first entry.
    if body.len() < 8 {
        return Err(BoxError::Truncated("stsd header"));
    }
    let entry_count = u32::from_be_bytes([body[4], body[5], body[6], body[7]]);
    if entry_count == 0 {
        return Err(BoxError::Malformed("stsd with no sample entries"));
    }
    let entries = &body[8..];
    let h = boxes::read_box_header(entries, 0)?;
    parse_sample_entry(&h, entries)
}

/// Parse one sample entry box (`h` within `data`) into a [`SampleEntry`]. Dispatches on the
/// four-CC: visual entries (§12.1.3) carry a 78-octet fixed header before width/height and
/// the codec-config child box; audio entries (§12.2.3) carry a 28-octet fixed header before
/// channelcount/samplerate.
fn parse_sample_entry(h: &BoxHeader, data: &[u8]) -> Result<SampleEntry, BoxError> {
    let kind = h.kind;
    let family = family_for(kind);
    let payload = h.body(data);
    let mut entry = SampleEntry {
        kind,
        family,
        width: 0,
        height: 0,
        sample_rate: 0,
        channels: 0,
        codec_head: Vec::new(),
        config_record: Vec::new(),
        reframer: Reframer::Passthrough,
    };
    if is_video_family(family) || is_nal_entry(&kind) {
        parse_visual_entry(payload, &mut entry)?;
    } else if family == "opus" || &kind == b"mp4a" {
        parse_audio_entry(payload, &mut entry)?;
    }
    // else: unknown `bytes` codec — no dimensions/params, passthrough. Still a valid pad.
    Ok(entry)
}

/// Parse a `VisualSampleEntry` (§12.1.3): a 6-octet reserved + 2-octet
/// data_reference_index (the SampleEntry base, §8.5.2.2), then 16 octets of pre-defined /
/// reserved, `width` u16, `height` u16, 14 more fixed octets, a 32-octet compressorname, a
/// 2-octet depth and a 2-octet pre-defined `-1`; child boxes (`avcC`, `hvcC`, `vpcC`,
/// `av1C`, `pasp`, …) follow. We read width/height and, for NAL codecs, the config record.
fn parse_visual_entry(payload: &[u8], entry: &mut SampleEntry) -> Result<(), BoxError> {
    // SampleEntry base (8) + pre_defined/reserved (16) = 24 octets before width (§12.1.3).
    const WIDTH_OFF: usize = 24;
    // width u16, height u16 at WIDTH_OFF; the fixed VisualSampleEntry fields run to 78.
    const CHILDREN_OFF: usize = 78;
    if payload.len() < CHILDREN_OFF {
        return Err(BoxError::Truncated("VisualSampleEntry fixed fields"));
    }
    entry.width = u16::from_be_bytes([payload[WIDTH_OFF], payload[WIDTH_OFF + 1]]) as u32;
    entry.height = u16::from_be_bytes([payload[WIDTH_OFF + 2], payload[WIDTH_OFF + 3]]) as u32;

    // The child boxes (codec-configuration + optional pasp/colr/…) begin at octet 78.
    let children = &payload[CHILDREN_OFF..];
    if is_nal_entry(&entry.kind) {
        let is_hevc = matches!(&entry.kind, b"hvc1" | b"hev1");
        let cfg_type: &[u8; 4] = if is_hevc { b"hvcC" } else { b"avcC" };
        // `avc3`/`hev1` carry parameter sets in-band; only `avc1`/`hvc1` guarantee an
        // out-of-band config record. Either way the samples are length-prefixed, so reframe.
        match boxes::find_child(children, boxes::fourcc(cfg_type))? {
            Some(cfg) => {
                // Retain the record verbatim for the passthrough/remux path regardless of
                // whether the Annex B conversion below accepts it.
                entry.config_record = cfg.body(children).to_vec();
                match nal_head_from_config(cfg.body(children), is_hevc) {
                    Ok((head, length_size)) => {
                        // For avc3/hev1 the head is present but redundant with the in-band
                        // sets; emitting it is harmless (duplicate parameter sets are ignored).
                        entry.codec_head = head;
                        entry.reframer = Reframer::Nal { length_size };
                    }
                    // A broken config record must not kill the demuxer: reframe with the
                    // default 4-octet length size and no head (decoder resyncs in-band).
                    Err(_) => entry.reframer = Reframer::Nal { length_size: 4 },
                }
            }
            // No config box (typical for avc3/hev1): length size defaults to 4 (§5.2.4.1.1
            // of 14496-15 makes lengthSizeMinusOne almost always 3), head stays empty.
            None => entry.reframer = Reframer::Nal { length_size: 4 },
        }
    }
    // vp09/av01: passthrough (raw frames); vpcC/av1C carry profile info the decoder
    // re-derives from the bitstream, so we do not need to parse them for enumeration.
    Ok(())
}

/// Parse an `AudioSampleEntry` (§12.2.3): the SampleEntry base (8), 8 octets reserved, a
/// `channelcount` u16, `samplesize` u16, 4 octets pre_defined/reserved, then `samplerate`
/// as a 16.16 fixed-point u32 (the integer part is the rate). We read channelcount and the
/// integer sample rate. For `mp4a` the nested `esds` box is then mined for the raw
/// **AudioSpecificConfig** ([`parse_esds`]): with one in hand the entry is promoted to the
/// `aac` family, the ASC becomes both the retained `config_record` (what a remux hands to
/// Matroska as A_AAC `CodecPrivate`, RFC 9559 §12) and the in-band `codec_head` (the
/// first-buffer-is-CodecPrivate contract the NAL families use). An absent or malformed
/// `esds` degrades to the `bytes` fallback — never an error (untrusted input). `Opus`'s
/// `dOps` is still not needed to enumerate samples (see `mp4/NOTES.md`).
fn parse_audio_entry(payload: &[u8], entry: &mut SampleEntry) -> Result<(), BoxError> {
    // 8 (base) + 8 (reserved) = 16 octets before channelcount; samplerate at octet 24.
    const CHANNELS_OFF: usize = 16;
    const RATE_OFF: usize = 24;
    // The fixed AudioSampleEntry runs to 28 octets; child boxes (`esds`, `dOps`) follow.
    const CHILDREN_OFF: usize = 28;
    if payload.len() < RATE_OFF + 4 {
        return Err(BoxError::Truncated("AudioSampleEntry fixed fields"));
    }
    entry.channels = u16::from_be_bytes([payload[CHANNELS_OFF], payload[CHANNELS_OFF + 1]]) as u32;
    // samplerate is a 16.16 fixed-point value; the high 16 bits are the integer Hz.
    entry.sample_rate = u32::from_be_bytes([
        payload[RATE_OFF],
        payload[RATE_OFF + 1],
        payload[RATE_OFF + 2],
        payload[RATE_OFF + 3],
    ]) >> 16;

    if &entry.kind == b"mp4a" {
        // Child-box offset: ISO-BMFF (14496-12) puts children right at octet 28, but a
        // QTFF-heritage SoundDescription (version at octet 8 — the first two octets of
        // 14496-12's "reserved" run) inserts 16 (v1) or 36 (v2) extra octets first.
        let version = u16::from_be_bytes([payload[8], payload[9]]);
        let children_off = match version {
            1 => CHILDREN_OFF + 16,
            2 => CHILDREN_OFF + 36,
            _ => CHILDREN_OFF,
        };
        // Every failure from here degrades to the `bytes` fallback (family untouched):
        // a broken esds must not kill track enumeration.
        let esds = payload
            .get(children_off..)
            .and_then(|children| boxes::find_child(children, boxes::fourcc(b"esds")).ok().flatten()
                .map(|h| h.body(children).to_vec()));
        if let Some((oti, asc)) = esds.as_deref().and_then(parse_esds) {
            // objectTypeIndication (14496-1 §7.2.6.6 Table 8): 0x40 = ISO/IEC 14496-3
            // (MPEG-4) Audio; 0x66/0x67/0x68 = MPEG-2 AAC Main/LC/SSR. Anything else in
            // an `mp4a` entry (e.g. 0x6B MP3) is not AAC — keep the `bytes` fallback.
            if matches!(oti, 0x40 | 0x66 | 0x67 | 0x68) && !asc.is_empty() {
                entry.family = "aac";
                entry.config_record = asc.clone();
                entry.codec_head = asc;
            }
        }
    }
    Ok(())
}

/// MPEG-4 descriptor tags (ISO/IEC 14496-1 §7.2.2.1 Table 1) the `esds` walk touches.
const TAG_ES_DESCR: u8 = 0x03;
const TAG_DECODER_CONFIG: u8 = 0x04;
const TAG_DECODER_SPECIFIC_INFO: u8 = 0x05;

/// Read one descriptor header at `data[at]` (ISO/IEC 14496-1 §8.3.3 expandable classes):
/// a tag octet, then the size as base-128 octets — bit 7 is the continuation flag, at
/// most 4 size octets. Returns the tag and the payload range, or `None` on truncation /
/// a >4-octet size (both malformed for our purposes — the caller falls back to `bytes`).
fn read_descriptor(data: &[u8], at: usize) -> Option<(u8, core::ops::Range<usize>)> {
    let tag = *data.get(at)?;
    let mut size = 0usize;
    let mut i = at + 1;
    for n in 0.. {
        let b = *data.get(i)?;
        i += 1;
        size = (size << 7) | (b & 0x7f) as usize;
        if b & 0x80 == 0 {
            break;
        }
        if n == 3 {
            return None; // §8.3.3 sizeOfInstance uses at most 4 octets
        }
    }
    let end = i.checked_add(size)?;
    if end > data.len() {
        return None;
    }
    Some((tag, i..end))
}

/// Extract `(objectTypeIndication, AudioSpecificConfig)` from an `esds` box payload.
///
/// Layout (ISO/IEC 14496-14 §5.6: the ES_Descriptor box is a FullBox wrapping exactly one
/// ES_Descriptor; ISO/IEC 14496-1 §7.2.6.5/§7.2.6.6):
/// - 4 octets FullBox version+flags, then the `ES_Descriptor` (tag 0x03):
///   `ES_ID` u16, a flags octet — `streamDependenceFlag`(0x80) adds a 2-octet
///   `dependsOn_ES_ID`, `URL_Flag`(0x40) adds `URLlength` + that many URL octets,
///   `OCRstreamFlag`(0x20) adds a 2-octet `OCR_ES_Id` — then nested descriptors;
/// - `DecoderConfigDescriptor` (tag 0x04): `objectTypeIndication` u8, then
///   streamType/upStream/reserved (1) + `bufferSizeDB` (3) + `maxBitrate` (4) +
///   `avgBitrate` (4) — 13 fixed octets — then nested descriptors;
/// - `DecoderSpecificInfo` (tag 0x05): for OTI 0x40 its payload **is** the raw
///   AudioSpecificConfig of ISO/IEC 14496-3 §1.6.2.1, returned verbatim (a remux keeps
///   it byte-exact — it becomes the Matroska A_AAC `CodecPrivate`, RFC 9559 §12).
///
/// `None` on any structural violation (the caller keeps the `bytes` fallback).
fn parse_esds(payload: &[u8]) -> Option<(u8, Vec<u8>)> {
    let (tag, es) = read_descriptor(payload, 4)?;
    if tag != TAG_ES_DESCR {
        return None;
    }
    let es_data = &payload[es];
    // ES_Descriptor fixed head: ES_ID (2) + flags/streamPriority (1).
    let flags = *es_data.get(2)?;
    let mut at = 3usize;
    if flags & 0x80 != 0 {
        at += 2; // dependsOn_ES_ID
    }
    if flags & 0x40 != 0 {
        at += 1 + *es_data.get(at)? as usize; // URLlength + URLstring
    }
    if flags & 0x20 != 0 {
        at += 2; // OCR_ES_Id
    }
    // Nested descriptors: find the DecoderConfigDescriptor.
    while at < es_data.len() {
        let (tag, r) = read_descriptor(es_data, at)?;
        if tag == TAG_DECODER_CONFIG {
            let dc = &es_data[r];
            let oti = *dc.first()?;
            // Past the 13 fixed octets: the nested DecoderSpecificInfo.
            let mut dat = 13usize;
            while dat < dc.len() {
                let (t, r) = read_descriptor(dc, dat)?;
                if t == TAG_DECODER_SPECIFIC_INFO {
                    return Some((oti, dc[r].to_vec()));
                }
                dat = r.end;
            }
            return None; // no DecoderSpecificInfo — no ASC to hand a remux
        }
        at = r.end;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_mapping_covers_the_box_types() {
        assert_eq!(family_for(*b"avc1"), "h264/annexb");
        assert_eq!(family_for(*b"avc3"), "h264/annexb");
        assert_eq!(family_for(*b"hvc1"), "h265/annexb");
        assert_eq!(family_for(*b"hev1"), "h265/annexb");
        assert_eq!(family_for(*b"vp09"), "vp9");
        assert_eq!(family_for(*b"av01"), "av1");
        assert_eq!(family_for(*b"Opus"), "opus");
        assert_eq!(family_for(*b"mp4a"), "bytes", "AAC: no decoder yet → bytes");
        assert_eq!(family_for(*b"twos"), "bytes", "unknown → bytes fallback");
    }

    /// Build a minimal `stsd` with one `avc1` visual entry carrying an `avcC` config, and
    /// check width/height + the reframed Annex B head come through.
    #[test]
    fn stsd_avc1_yields_dims_and_annex_b_head() {
        let sps = [0x67u8, 0x42, 0x00, 0x0A];
        let pps = [0x68u8, 0xCE, 0x3C, 0x80];
        // avcC body (ISO/IEC 14496-15 §5.3.3.1): version, profile/compat/level, lengthSize,
        // one SPS, one PPS.
        let mut avcc = vec![1u8, 0x42, 0x00, 0x0A, 0xFF, 0xE1];
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&sps);
        avcc.push(1);
        avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&pps);
        let avcc_box = box_bytes(b"avcC", &avcc);

        // VisualSampleEntry: 78 fixed octets with width/height at 24, then the avcC child.
        let mut ve = vec![0u8; 78];
        ve[24..26].copy_from_slice(&640u16.to_be_bytes());
        ve[26..28].copy_from_slice(&480u16.to_be_bytes());
        ve.extend_from_slice(&avcc_box);
        let avc1_box = box_bytes(b"avc1", &ve);

        // stsd payload: FullBox head + entry_count(1) + the avc1 entry.
        let mut stsd = vec![0u8; 4];
        stsd.extend_from_slice(&1u32.to_be_bytes());
        stsd.extend_from_slice(&avc1_box);

        let entry = parse_stsd(&stsd).unwrap();
        assert_eq!(entry.kind, *b"avc1");
        assert_eq!(entry.family, "h264/annexb");
        assert_eq!((entry.width, entry.height), (640, 480));
        assert!(matches!(entry.reframer, Reframer::Nal { length_size: 4 }));
        // The head is SPS then PPS as 4-byte start-code NALs.
        let mut want = Vec::new();
        for nal in [&sps[..], &pps[..]] {
            want.extend_from_slice(&[0, 0, 0, 1]);
            want.extend_from_slice(nal);
        }
        assert_eq!(entry.codec_head, want);
    }

    /// A `vp09` visual entry is passthrough with dimensions and no head.
    #[test]
    fn stsd_vp09_is_passthrough_with_dims() {
        let mut ve = vec![0u8; 78];
        ve[24..26].copy_from_slice(&128u16.to_be_bytes());
        ve[26..28].copy_from_slice(&96u16.to_be_bytes());
        let vp09_box = box_bytes(b"vp09", &ve);
        let mut stsd = vec![0u8; 4];
        stsd.extend_from_slice(&1u32.to_be_bytes());
        stsd.extend_from_slice(&vp09_box);
        let entry = parse_stsd(&stsd).unwrap();
        assert_eq!(entry.family, "vp9");
        assert_eq!((entry.width, entry.height), (128, 96));
        assert!(matches!(entry.reframer, Reframer::Passthrough));
        assert!(entry.codec_head.is_empty());
    }

    /// An `mp4a` audio entry yields channels + rate and — without an `esds` — stays on
    /// the `bytes` fallback family (no ASC, nothing to hand a remux as CodecPrivate).
    #[test]
    fn stsd_mp4a_audio_params() {
        let mut ae = vec![0u8; 28];
        ae[16..18].copy_from_slice(&2u16.to_be_bytes()); // channelcount
        ae[24..28].copy_from_slice(&(48_000u32 << 16).to_be_bytes()); // 16.16 samplerate
        let mp4a_box = box_bytes(b"mp4a", &ae);
        let mut stsd = vec![0u8; 4];
        stsd.extend_from_slice(&1u32.to_be_bytes());
        stsd.extend_from_slice(&mp4a_box);
        let entry = parse_stsd(&stsd).unwrap();
        assert_eq!(entry.family, "bytes");
        assert_eq!((entry.channels, entry.sample_rate), (2, 48_000));
        assert!(entry.config_record.is_empty(), "no esds → no ASC");
    }

    /// Serialise an `esds` box body: FullBox head + ES_Descriptor(0x03) wrapping
    /// DecoderConfigDescriptor(0x04) wrapping DecoderSpecificInfo(0x05) = `asc`
    /// (ISO/IEC 14496-14 §5.6; 14496-1 §7.2.6.5/.6). `long_sizes` uses the 4-octet
    /// base-128 form (`0x80 0x80 0x80 nn`) every real-world encoder loves to emit.
    fn esds_body(oti: u8, asc: &[u8], long_sizes: bool) -> Vec<u8> {
        let size = |v: &mut Vec<u8>, n: usize| {
            if long_sizes {
                v.extend_from_slice(&[0x80, 0x80, 0x80, n as u8]);
            } else {
                v.push(n as u8);
            }
        };
        // DecoderSpecificInfo (tag 0x05): payload = the raw ASC.
        let mut dsi = vec![TAG_DECODER_SPECIFIC_INFO];
        size(&mut dsi, asc.len());
        dsi.extend_from_slice(asc);
        // DecoderConfigDescriptor (tag 0x04): oti, streamType(0x15 = audio<<2|1),
        // bufferSizeDB(3), maxBitrate(4), avgBitrate(4), then the DSI.
        let mut dc_payload = vec![oti, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        dc_payload.extend_from_slice(&dsi);
        let mut dc = vec![TAG_DECODER_CONFIG];
        size(&mut dc, dc_payload.len());
        dc.extend_from_slice(&dc_payload);
        // ES_Descriptor (tag 0x03): ES_ID(2) + flags(1), then the DecoderConfig.
        let mut es_payload = vec![0, 1, 0];
        es_payload.extend_from_slice(&dc);
        let mut es = vec![TAG_ES_DESCR];
        size(&mut es, es_payload.len());
        es.extend_from_slice(&es_payload);
        // esds is a FullBox: version(1) + flags(3) before the descriptor.
        let mut body = vec![0u8; 4];
        body.extend_from_slice(&es);
        body
    }

    /// An `mp4a` sample entry with a 28-octet fixed part (§12.2.3: base 8 + reserved 8 +
    /// channelcount@16 + samplesize@18 + pre_defined/reserved@20 + 16.16 samplerate@24)
    /// followed by an `esds` child at octet 28.
    fn mp4a_entry(esds: &[u8], rate: u32, channels: u16) -> Vec<u8> {
        let mut ae = vec![0u8; 28];
        ae[16..18].copy_from_slice(&channels.to_be_bytes());
        ae[24..28].copy_from_slice(&(rate << 16).to_be_bytes());
        ae.extend_from_slice(&box_bytes(b"esds", esds));
        ae
    }

    /// The `esds` walk: ES_Descriptor(0x03) → DecoderConfigDescriptor(0x04, OTI 0x40) →
    /// DecoderSpecificInfo(0x05) = the raw AudioSpecificConfig, promoted to family `aac`
    /// with the ASC as both `config_record` and the in-band `codec_head`.
    #[test]
    fn stsd_mp4a_with_esds_promotes_to_aac() {
        // ASC 0x11 0x90: AOT=2 (AAC-LC, 5 bits), freq index 3 = 48 kHz (4 bits),
        // channelConfiguration 2 (4 bits) — ISO/IEC 14496-3 §1.6.2.1.
        let asc = [0x11u8, 0x90];
        let ae = mp4a_entry(&esds_body(0x40, &asc, false), 48_000, 2);
        let mut stsd = vec![0u8; 4];
        stsd.extend_from_slice(&1u32.to_be_bytes());
        stsd.extend_from_slice(&box_bytes(b"mp4a", &ae));
        let entry = parse_stsd(&stsd).unwrap();
        assert_eq!(entry.family, "aac");
        assert_eq!((entry.channels, entry.sample_rate), (2, 48_000));
        assert_eq!(entry.config_record, asc, "CodecPrivate = the ASC verbatim");
        assert_eq!(entry.codec_head, asc, "in-band head = the ASC (first-buffer contract)");
        assert!(matches!(entry.reframer, Reframer::Passthrough), "raw AAC AUs, no reframe");
    }

    /// The 4-octet base-128 expandable sizes (`0x80 0x80 0x80 nn`, 14496-1 §8.3.3) parse
    /// identically to the 1-octet form.
    #[test]
    fn esds_long_expandable_sizes_parse() {
        let asc = [0x12u8, 0x08]; // AOT=2, freq index 4 = 44.1 kHz, 1 channel
        let ae = mp4a_entry(&esds_body(0x40, &asc, true), 44_100, 1);
        let mut stsd = vec![0u8; 4];
        stsd.extend_from_slice(&1u32.to_be_bytes());
        stsd.extend_from_slice(&box_bytes(b"mp4a", &ae));
        let entry = parse_stsd(&stsd).unwrap();
        assert_eq!(entry.family, "aac");
        assert_eq!(entry.config_record, asc);
    }

    /// A non-AAC objectTypeIndication (0x6B = MPEG-1/2 Layer III in `mp4a`) keeps the
    /// `bytes` fallback — raw MP3 frames are not Matroska A_AAC material.
    #[test]
    fn esds_non_aac_oti_stays_bytes() {
        let ae = mp4a_entry(&esds_body(0x6B, &[0x11, 0x90], false), 44_100, 2);
        let mut stsd = vec![0u8; 4];
        stsd.extend_from_slice(&1u32.to_be_bytes());
        stsd.extend_from_slice(&box_bytes(b"mp4a", &ae));
        let entry = parse_stsd(&stsd).unwrap();
        assert_eq!(entry.family, "bytes");
        assert!(entry.config_record.is_empty());
    }

    /// A malformed esds (truncated descriptors, garbage sizes) degrades to `bytes`,
    /// never an error or panic (untrusted input).
    #[test]
    fn esds_malformed_degrades_to_bytes() {
        for bad in [
            &[0u8; 4][..],                        // FullBox head only, no descriptor
            &[0, 0, 0, 0, 0x03][..],              // tag with no size octet
            &[0, 0, 0, 0, 0x03, 0x80][..],        // dangling continuation bit
            &[0, 0, 0, 0, 0x03, 0x7F][..],        // size runs past the payload
            &[0, 0, 0, 0, 0x04, 0x02, 0x40, 0][..], // DecoderConfig without ES wrapper
        ] {
            let ae = mp4a_entry(bad, 48_000, 2);
            let mut stsd = vec![0u8; 4];
            stsd.extend_from_slice(&1u32.to_be_bytes());
            stsd.extend_from_slice(&box_bytes(b"mp4a", &ae));
            let entry = parse_stsd(&stsd).unwrap();
            assert_eq!(entry.family, "bytes", "bad esds {bad:02X?} must fall back");
            assert!(entry.config_record.is_empty());
        }
    }

    #[test]
    fn malformed_stsd_errors_not_panic() {
        assert!(parse_stsd(&[]).is_err());
        assert!(parse_stsd(&[0, 0, 0, 0, 0, 0, 0, 0]).is_err(), "entry_count 0");
        // entry_count 1 but a truncated entry box.
        let mut stsd = vec![0u8; 4];
        stsd.extend_from_slice(&1u32.to_be_bytes());
        stsd.extend_from_slice(&[0, 0, 0, 8]); // a box header cut short
        assert!(parse_stsd(&stsd).is_err());
    }

    fn box_bytes(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((8 + body.len()) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }
}
