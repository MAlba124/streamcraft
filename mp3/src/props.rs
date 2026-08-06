//! No-decode MP3 property probing: sample rate, channel count, bitrate and — the reason
//! this module exists — **duration**, from the first few KiB of audio.
//!
//! An MP3 file has no header that states its length. A CBR stream's duration follows from
//! its byte count and bitrate, but a VBR stream's does not, and neither can be had by
//! decoding without reading the whole file. Encoders therefore hide a summary inside the
//! *first* frame's payload — where a decoder sees only silence, because the frame carries
//! no audio: **Xing/Info** (the Xing SDK layout LAME adopted) or **VBRI** (Fraunhofer's).
//! Either one gives an exact frame count, hence an exact duration, from one frame's worth
//! of bytes; without one, the first frame's bitrate gives an estimate. LAME's extension to
//! the Xing layout goes one better and states how much of that coded length is the
//! encoder's own delay and padding rather than audio, which is subtracted here.
//!
//! The same header answers the *other* question a file with no index can't otherwise
//! answer — which byte is 3 minutes in — through its optional 100-point seek table
//! ([`XingToc`]). Duration and that table together are everything a player needs to offer
//! a seek bar over a bare VBR MP3.
//!
//! Like [`crate::id3`], this is a free function over a `&[u8]` window — no IO, no pipeline
//! types, no allocation. It reuses `mp3dec`'s framer (`next_frame`) rather than forking a
//! second header parser, so "what counts as a frame here" is exactly what the decoder
//! believes: a length-derivable header confirmed by the next frame's syncword landing at
//! its end.
//!
//! The frame header fields themselves are ISO/IEC 11172-3 §2.4.1.3 / §2.4.2.3, parsed by
//! `oxideav-mp3`'s `parse_header`.

use oxideav_mp3::frame::{parse_header, Mp3FrameHeader};

use crate::mp3dec::{next_frame, Framed};

/// The **Xing/Info seek table** (the Xing SDK's VBR header, flags bit 2): 100 bytes that
/// map time to byte position across the stream, which is the only way to seek a VBR MP3
/// without walking every frame header.
///
/// Entry `i` answers "how far into the stream's *bytes* is the point `i`/100 of the way
/// through its *duration*", as a fraction in 256ths — so `points[0]` is 0 and `points[99]`
/// is a little under 256. The Xing SDK's own `seek_point()` reads it as
/// `byte = (points[i] / 256) × stream_bytes`, and that is exactly [`byte_at`](Self::byte_at)
/// (ffmpeg's `read_xing_toc` in `libavformat/mp3dec.c` does the same rescale). The table is
/// as accurate as the encoder made it — a hundred points over the whole stream, so a
/// landing is within ~1 % of the target and the decoder resyncs to the next frame header
/// from there.
///
/// `stream_bytes` is the length the fractions are relative to: the byte count the encoder
/// wrote in the header's own size field ([`AudioProps::stream_bytes`]) when it wrote one,
/// else the audio region's length (the file minus its leading and trailing tags). The two
/// agree unless a tagger appended something after the encoder finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XingToc {
    /// The hundred raw table bytes, each a position in 256ths of `stream_bytes`.
    pub points: [u8; XingToc::LEN],
}

impl XingToc {
    /// The table is a fixed hundred points (one per percent of the duration).
    pub const LEN: usize = 100;

    /// Byte offset of seek point `i`, **relative to the first byte of the first audio
    /// frame**, over a stream of `stream_bytes` bytes: `points[i] / 256 × stream_bytes`.
    /// `None` for `i >= 100`. The arithmetic is done in `u128` and cannot overflow or
    /// divide by zero, so no input here can panic.
    pub fn byte_at(&self, i: usize, stream_bytes: u64) -> Option<u64> {
        let frac = u128::from(*self.points.get(i)?);
        Some((frac * u128::from(stream_bytes) / 256).min(u128::from(u64::MAX)) as u64)
    }

    /// Stream time of seek point `i` — `i`/100 of `duration_ns`, the other half of the
    /// mapping. `None` for `i >= 100`.
    pub fn time_at(&self, i: usize, duration_ns: u64) -> Option<u64> {
        if i >= Self::LEN {
            return None;
        }
        Some((i as u128 * u128::from(duration_ns) / Self::LEN as u128) as u64)
    }
}

/// What an MP3 stream declares about itself, read from its first frame (plus whatever VBR
/// header that frame carries).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioProps {
    /// Stream duration in nanoseconds, or `None` when it cannot be derived at all (a
    /// free-format stream with no VBR header states neither a frame count nor a bitrate).
    pub duration_ns: Option<u64>,
    /// `true` when `duration_ns` came from a Xing/Info or VBRI **frame count** — an exact
    /// figure, less the encoder delay and padding when a LAME extension states them, so it
    /// is the audio's length and not the coded stream's. `false` when it is the
    /// constant-bitrate estimate, which is off by whatever the real stream's bitrate
    /// variation and trailing metadata amount to.
    pub exact: bool,
    /// Sampling frequency in Hz (ISO/IEC 11172-3 §2.4.2.3 table).
    pub sample_rate: u32,
    /// Channel count: 1 for single_channel, 2 for every other mode (§2.4.2.7 `nch`).
    pub channels: u32,
    /// The **first frame's** bitrate in kbit/s, or `None` for free format. For a VBR
    /// stream this is that one frame's bitrate, not the file's average — the frame count
    /// is the figure to trust there, and it is what `duration_ns` uses.
    pub bitrate_kbps: Option<u32>,
    /// Byte offset of the first frame's sync word **within the `audio` window that was
    /// probed** — normally 0, but taggers leave junk between a tag and the first sync
    /// word. Add it to wherever that window started and you have the file offset the
    /// audio really begins at, which is the anchor every byte figure below is relative to.
    pub first_frame_offset: usize,
    /// The [`XingToc`] this stream's VBR header carries, if it carries one (Xing/Info
    /// flags bit 2). Absent for a stream with no VBR header, one whose header omits the
    /// table, and for VBRI — whose own seek table has a different, encoder-specific layout
    /// that is not read here.
    pub toc: Option<XingToc>,
    /// The stream's byte count as the VBR header itself states it (Xing/Info flags bit 1,
    /// or VBRI's fixed field) — the length [`XingToc::byte_at`] fractions are relative to.
    /// `None` when the header omits it (or there is no header), and then the caller's own
    /// measurement of the audio region is the best available substitute.
    pub stream_bytes: Option<u32>,
}

/// Probe an MP3's properties without decoding it.
///
/// `audio` is a window starting at the **first post-tag byte** — a few KiB is plenty, and
/// it must be long enough to hold the whole first frame (at most ~1440 bytes) for the VBR
/// header inside it to be found. `audio_len` is the length of the audio itself: the file
/// size minus the leading ID3v2 tag ([`v2_total_len`](crate::id3::v2_total_len)) and any
/// trailing APEv2 ([`ape_tail_len`](crate::id3::ape_tail_len)) / ID3v1 blocks. It is only
/// used for the CBR estimate, where counting tag bytes as audio would inflate the duration.
///
/// `None` when no MPEG audio frame can be found in `audio` at all.
pub fn probe_props(audio: &[u8], audio_len: u64) -> Option<AudioProps> {
    // Reuse the decoder's framer so the notion of "the first frame" is identical on both
    // paths. `NeedMore` means a valid header was found but its frame overruns the window;
    // the header alone still yields rate/channels/bitrate (and the CBR estimate), just not
    // the VBR header buried in the frame's payload.
    let (offset, header) = match next_frame(audio) {
        Framed::Frame { offset, header, .. } => (offset, header),
        Framed::NeedMore { offset } => (offset, parse_header(audio.get(offset..)?).ok()?),
        Framed::NoSync => return None,
    };
    if header.sample_rate_hz == 0 {
        return None;
    }
    let mut props = AudioProps {
        duration_ns: None,
        exact: false,
        sample_rate: header.sample_rate_hz,
        channels: u32::from(header.channel_count()),
        bitrate_kbps: header.bitrate_kbps,
        first_frame_offset: offset,
        toc: None,
        stream_bytes: None,
    };

    // The VBR header lives inside the first frame, so it needs the frame's whole payload.
    let frame = header
        .frame_len()
        .and_then(|len| offset.checked_add(len))
        .and_then(|end| audio.get(offset..end));
    if let Some(vbr) = frame.and_then(|f| vbr_header(f, &header)) {
        props.toc = vbr.toc;
        props.stream_bytes = vbr.bytes;
        // Exact: `frames` whole frames of `samples_per_frame` samples each (1152 for
        // MPEG-1 Layer III, 576 for the MPEG-2 / MPEG-2.5 low-sampling-rate variant —
        // ISO/IEC 13818-3 §2.4.2.1) at the header's sampling frequency.
        let coded = u64::from(vbr.frames) * u64::from(header.samples_per_frame());
        // …of which the first `delay` and the last `padding` samples are not audio: an MP3
        // stream can only be a whole number of frames long, so the encoder prepends its own
        // filter delay and appends however much silence the last frame needs, then records
        // both in the LAME extension so a player can hand back the original sample count.
        // Dropping them is what makes a gapless album gapless, and it is the length every
        // other tool reports — see [`lame_trim`] for the arithmetic ffmpeg does. Still
        // `exact`: both counts are stated by the encoder, nothing here is estimated.
        let trim = u64::from(vbr.trim);
        let samples = if trim < coded { coded - trim } else { coded };
        let ns = u128::from(samples) * 1_000_000_000 / u128::from(header.sample_rate_hz);
        props.duration_ns = Some(ns.min(u128::from(u64::MAX)) as u64);
        props.exact = true;
        return Some(props);
    }

    // No VBR header: assume the first frame's bitrate holds throughout, which is exactly
    // what a CBR stream guarantees (§2.4.2.3 — every frame repeats the bitrate_index).
    if let Some(kbps) = header.bitrate_kbps.filter(|&k| k > 0) {
        // bytes × 8 bits ÷ (kbps × 1000 bit/s) × 1e9 ns/s, folded to avoid the rounding of
        // an intermediate division.
        let ns = u128::from(audio_len) * 8_000_000 / u128::from(kbps);
        props.duration_ns = Some(ns.min(u128::from(u64::MAX)) as u64);
    }
    Some(props)
}

/// What a VBR header states about the stream's length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Vbr {
    /// Total frames in the stream, including the header frame itself.
    frames: u32,
    /// Samples of non-audio to drop from the two ends — encoder delay plus trailing
    /// padding, from the LAME extension. `0` when there is no LAME extension to read them
    /// from, which is the pre-gapless behaviour: report the whole coded length.
    trim: u32,
    /// The stream's byte count, when the header states one (Xing/Info flags bit 1).
    bytes: Option<u32>,
    /// The 100-byte seek table, when the header carries one (Xing/Info flags bit 2).
    toc: Option<XingToc>,
}

/// The stream length declared by the VBR header inside `frame`, if it carries one.
///
/// Two mutually exclusive layouts occupy the first frame's otherwise-unused payload:
///
/// * **Xing/Info** — the Xing SDK's VBR header, which LAME writes and every decoder reads.
///   It begins immediately after the frame's side information, so its offset depends on
///   the header: 4 header bytes, 2 more if the CRC-check word is present (ISO/IEC 11172-3
///   §2.4.1), then the Layer III side-info block (§2.4.1.7), whose width is fixed per
///   version and channel count — MPEG-1: 17 bytes mono / 32 stereo; MPEG-2 and MPEG-2.5:
///   9 / 17. The tag is `"Xing"` (VBR) or `"Info"` (CBR, written for the TOC and gapless
///   fields), then a 32-bit big-endian flags word; bit 0 means a 32-bit big-endian frame
///   count follows immediately, bit 1 a byte count, bit 2 a 100-byte seek TOC, bit 3 a
///   quality indicator. Whatever those flags announce is followed by LAME's extension —
///   see [`lame_trim`].
/// * **VBRI** — Fraunhofer's equivalent, written by their encoder only. It ignores the
///   side info and sits at a fixed 32 bytes past the 4-byte frame header, laid out as
///   `"VBRI"`, a 16-bit version, a 16-bit delay, a 16-bit quality, a 32-bit byte count and
///   a 32-bit **frame count** — the field at offset 14. Its delay field is a byte count
///   into the stream, not a sample count to trim, so nothing is trimmed on this path.
fn vbr_header(frame: &[u8], header: &Mp3FrameHeader) -> Option<Vbr> {
    let side_info = match (header.version.is_lsf(), header.channel_count()) {
        (false, 1) => 17,
        (false, _) => 32,
        (true, 1) => 9,
        (true, _) => 17,
    };
    let at = 4 + usize::from(header.crc_protected) * 2 + side_info;
    if let Some(xing) = frame.get(at..) {
        if xing.starts_with(b"Xing") || xing.starts_with(b"Info") {
            let flags = be32(xing.get(4..8)?);
            if flags & 0x1 == 0 {
                return None;
            }
            // The optional fields are stored in flag order and are variable-width, so the
            // only way to the ones behind is to walk them: frame count (bit 0, required
            // above), byte count (bit 1), the 100-byte seek TOC (bit 2), quality (bit 3),
            // and then LAME's extension. A field the flags do not announce is simply not
            // there — it is not zero-filled — which is why this is a walk and not a table
            // of fixed offsets.
            let frames = be32(xing.get(8..12)?);
            let mut field = 12;
            let mut bytes = None;
            if flags & 0x2 != 0 {
                bytes = xing.get(field..field + 4).map(be32);
                field += 4;
            }
            let mut toc = None;
            if flags & 0x4 != 0 {
                toc = xing
                    .get(field..field + XingToc::LEN)
                    .and_then(|t| <[u8; XingToc::LEN]>::try_from(t).ok())
                    .map(|points| XingToc { points });
                field += XingToc::LEN;
            }
            if flags & 0x8 != 0 {
                field += 4; // the quality indicator, which tells a player nothing
            }
            let trim = xing.get(field..).and_then(lame_trim).unwrap_or(0);
            return Some(Vbr { frames, trim, bytes, toc });
        }
    }
    let vbri = frame.get(36..)?;
    if !vbri.starts_with(b"VBRI") {
        return None;
    }
    // VBRI's own seek table follows the fixed fields, but its shape is declared by three
    // more header fields (entry count, scale, and a per-entry width of 1–4 bytes) rather
    // than fixed like Xing's, and Fraunhofer's encoder is rare enough that reading it has
    // never been needed here — the byte count and frame count are taken, the table is not.
    Some(Vbr {
        frames: be32(vbri.get(14..18)?),
        trim: 0,
        bytes: vbri.get(10..14).map(be32),
        toc: None,
    })
}

/// Encoder delay + trailing padding, in samples, from the LAME extension at `ext` — the
/// gapless-playback fields that turn a coded frame count into the original sample count.
///
/// Layout is Gabriel Bouvigne's "MP3 Info Tag" specification (mp3-tech.org's LAME header
/// description, revision 1), whose 36 bytes follow the Xing/Info fields: a 9-byte encoder
/// short version string (`"LAME3.99r"`, `"Lavf61.7.100"` truncated, …), then revision and
/// VBR method, lowpass, the three ReplayGain fields, encoding flags/ATH type and a bitrate
/// byte — putting the delays at **0x15**, three bytes packing two 12-bit counts: the
/// encoder delay (samples of the encoder's own filter warm-up prepended to the stream)
/// then the padding (silence appended so the last frame is whole).
///
/// Only LAME's own string and ffmpeg's two (`Lavc`/`Lavf`, written by the same tag writer)
/// are trusted, because those three are exactly the writers whose bytes are known to be at
/// these offsets; anything else keeps a random 24 bits from being read as a length. This
/// is ffmpeg's rule too, in `libavformat/mp3dec.c` (`mp3_parse_info_tag`).
///
/// **The 529-sample decoder delay is deliberately not in the sum.** ffmpeg trims the head
/// at `delay + 528 + 1` and stops at `frames × spf − padding + 528 + 1`; the constant is
/// the *decoder's* own reconstruction delay, present at both ends, so it cancels and the
/// playable length is exactly `frames × spf − delay − padding`. Adding it would make every
/// duration 12 ms short of ffprobe's — checked against ffprobe over 400-odd real files and
/// against LAME- and Lavf-written test signals of known length, where this sum lands on the
/// nominal duration exactly.
///
/// `None` when the extension is absent, unrecognised or truncated: the caller then reports
/// the untrimmed coded length, as it always did.
fn lame_trim(ext: &[u8]) -> Option<u32> {
    let writer = ext.get(..4)?;
    if writer != b"LAME" && writer != b"Lavc" && writer != b"Lavf" {
        return None;
    }
    let d = ext.get(0x15..0x18)?;
    let delay = (u32::from(d[0]) << 4) | u32::from(d[1] >> 4);
    let padding = (u32::from(d[1] & 0x0F) << 8) | u32::from(d[2]);
    // No range check is needed on either: 12 bits cannot exceed 4095, comfortably under the
    // 1152 (or 576) samples of one frame that a padding count is meant to fill and the two
    // frames of delay LAME can ask for. That the two together are shorter than the stream
    // is the caller's check, since only it knows the frame count.
    Some(delay + padding)
}

/// Big-endian `u32`, 0 on a short slice (total by design — probing never panics).
fn be32(b: &[u8]) -> u32 {
    b.get(..4).map_or(0, |s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}
