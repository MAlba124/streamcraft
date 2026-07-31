//! Hand-built minimal files — WAV, FLAC, MP3, MP4/M4A, Ogg (Opus, Vorbis, FLAC) — byte for
//! byte from their specifications.
//!
//! In the library rather than in `tests/` because three callers need the same definition —
//! the unit tests, the integration tests, and the allocation harness in
//! `examples/scan_alloc_check.rs` — and a fixture that drifts between them is a test that
//! silently stops testing. (Same reasoning as `profluens_core::harness`.)
//!
//! Nothing here is committed as a binary and nothing shells out to `ffmpeg`: every byte is
//! written from the specification cited at the builder, which is what makes a failing test
//! point at a parser rather than at whatever an encoder happened to emit that year.
//!
//! These are *builders*, not fast paths: they allocate freely.

#![allow(clippy::disallowed_methods)] // fixture builders — one-time, test/harness-only (spec: allocation discipline)

// --- RIFF/WAVE (Multimedia Programming Interface and Data Specifications 1.0, 1991) ------

/// One RIFF chunk: `<ckID:4> <u32 ckSize> <data>` plus the pad byte an odd size requires
/// (§"RIFF Chunks").
pub fn riff_chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = id.to_vec();
    v.extend_from_slice(&(body.len() as u32).to_le_bytes());
    v.extend_from_slice(body);
    if body.len() % 2 == 1 {
        v.push(0);
    }
    v
}

/// A `fmt ` chunk for uncompressed PCM (§"WAVE Format Chunk").
pub fn fmt_chunk(rate: u32, channels: u16, bits: u16) -> Vec<u8> {
    let block = channels * bits / 8;
    let mut b = Vec::new();
    b.extend_from_slice(&1u16.to_le_bytes()); // WAVE_FORMAT_PCM
    b.extend_from_slice(&channels.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * u32::from(block)).to_le_bytes()); // nAvgBytesPerSec
    b.extend_from_slice(&block.to_le_bytes());
    b.extend_from_slice(&bits.to_le_bytes());
    riff_chunk(b"fmt ", &b)
}

/// A `LIST`/`INFO` chunk (§"INFO List Chunk"). `entries` are `(4-character INFO id, value)`,
/// each stored as a NUL-terminated string.
pub fn info_chunk(entries: &[(&str, &str)]) -> Vec<u8> {
    let mut b = b"INFO".to_vec();
    for (id, text) in entries {
        let mut key = [b' '; 4];
        for (i, c) in id.bytes().take(4).enumerate() {
            key[i] = c;
        }
        let mut z = text.as_bytes().to_vec();
        z.push(0);
        b.extend_from_slice(&riff_chunk(&key, &z));
    }
    riff_chunk(b"LIST", &b)
}

/// Wrap chunks in the WAVE form: `"RIFF" <u32 size> "WAVE" <chunk>*` (§"RIFF Form").
pub fn riff(chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut body = b"WAVE".to_vec();
    for c in chunks {
        body.extend_from_slice(c);
    }
    let mut f = b"RIFF".to_vec();
    f.extend_from_slice(&(body.len() as u32).to_le_bytes());
    f.extend_from_slice(&body);
    f
}

/// A complete PCM WAV of `frames` sample frames. `tail` puts the `LIST`/`INFO` chunk *after*
/// the `data` chunk — the layout a tagger that appends produces, and the one that costs the
/// scanner a second positioned read.
pub fn wav(
    rate: u32,
    channels: u16,
    bits: u16,
    frames: usize,
    info: &[(&str, &str)],
    tail: bool,
) -> Vec<u8> {
    let block = usize::from(channels * bits / 8);
    // A cheap triangle, so the payload is not a compressible run of zeros.
    let data: Vec<u8> = (0..frames * block).map(|i| (i % 251) as u8).collect();
    let mut chunks = vec![fmt_chunk(rate, channels, bits)];
    if info.is_empty() {
        chunks.push(riff_chunk(b"data", &data));
    } else if tail {
        chunks.push(riff_chunk(b"data", &data));
        chunks.push(info_chunk(info));
    } else {
        chunks.push(info_chunk(info));
        chunks.push(riff_chunk(b"data", &data));
    }
    riff(&chunks)
}

// --- AIFF / AIFF-C (Apple, Audio Interchange File Format 1.3 + the 1991 AIFF-C addendum) --

/// One AIFF chunk: `<ckID:4> <u32 ckSize> <data>` plus the pad byte an odd size requires
/// ("File Structure"). Same grammar as RIFF, **big-endian** size.
pub fn aiff_chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = id.to_vec();
    v.extend_from_slice(&(body.len() as u32).to_be_bytes());
    v.extend_from_slice(body);
    if body.len() % 2 == 1 {
        v.push(0);
    }
    v
}

/// Wrap chunks in a FORM: `"FORM" <u32 ckSize> <formType> <chunk>*` ("File Structure").
/// `form_type` is `b"AIFF"` or `b"AIFC"`.
pub fn form(form_type: &[u8; 4], chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut body = form_type.to_vec();
    for c in chunks {
        body.extend_from_slice(c);
    }
    let mut f = b"FORM".to_vec();
    f.extend_from_slice(&(body.len() as u32).to_be_bytes());
    f.extend_from_slice(&body);
    f
}

/// An 80-bit IEEE 754 extended ("SANE Extended") big-endian float, built from an `f64` by
/// re-biasing the exponent and making the leading integer bit explicit.
///
/// binary64 is sign(1) + exponent(11, bias 1023) + fraction(52) with an *implicit* leading
/// one; the 80-bit format is sign(1) + exponent(15, bias 16383) + a 64-bit mantissa whose
/// bit 63 is that leading one, written out. So the conversion is a bias swap and an 11-bit
/// left shift of the fraction — exact for every value binary64 can hold, which is every
/// sample rate in use.
pub fn extended80(value: f64) -> [u8; 10] {
    let mut out = [0u8; 10];
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i32;
    // Zero and denormals both encode as all-zero here; neither is a sample rate.
    if exponent == 0 {
        return out;
    }
    let sign = ((bits >> 63) & 1) as u8;
    let fraction = bits & 0x000F_FFFF_FFFF_FFFF;
    let exp80 = (exponent - 1023 + 16383) as u16;
    let mantissa = (1u64 << 63) | (fraction << 11);
    out[0] = (sign << 7) | ((exp80 >> 8) as u8 & 0x7F);
    out[1] = (exp80 & 0xFF) as u8;
    out[2..10].copy_from_slice(&mantissa.to_be_bytes());
    out
}

/// A `COMM` chunk ("Common Chunk"): channels, sample frames, sample size, and the 80-bit
/// extended sample rate. `compression` adds the AIFF-C `compressionType` and an empty Pascal
/// `compressionName`, which is what makes the chunk 22+ bytes rather than 18.
pub fn comm_chunk(
    rate: u32,
    channels: u16,
    bits: u16,
    frames: u32,
    compression: Option<[u8; 4]>,
) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&channels.to_be_bytes());
    b.extend_from_slice(&frames.to_be_bytes());
    b.extend_from_slice(&bits.to_be_bytes());
    b.extend_from_slice(&extended80(f64::from(rate)));
    if let Some(kind) = compression {
        b.extend_from_slice(&kind);
        b.push(0); // an empty pstring: a length octet of zero
    }
    aiff_chunk(b"COMM", &b)
}

/// A four-character chunk id from a `&str`, space-padded — so a fixture can spell `"(c) "`
/// and `"NAME"` alike.
fn chunk_id(name: &str) -> [u8; 4] {
    let mut out = [b' '; 4];
    for (i, c) in name.bytes().take(4).enumerate() {
        out[i] = c;
    }
    out
}

/// A complete AIFF (`form_type` = `b"AIFF"`) or AIFF-C (`b"AIFC"`) file: `COMM`, the text
/// chunks, and an `SSND` body so the file is not pure metadata.
///
/// `text` entries are `(chunk id, value)` — `NAME`, `AUTH`, `ANNO`, `(c) `.
pub fn aiff(
    form_type: &[u8; 4],
    rate: u32,
    channels: u16,
    bits: u16,
    frames: u32,
    text: &[(&str, &str)],
    compression: Option<[u8; 4]>,
) -> Vec<u8> {
    let mut chunks = vec![comm_chunk(rate, channels, bits, frames, compression)];
    for (id, value) in text {
        chunks.push(aiff_chunk(&chunk_id(id), value.as_bytes()));
    }
    // "offset" and "blockSize" precede the samples in a Sound Data Chunk; the payload itself
    // is never read by a tag scan.
    let block = usize::from(channels * bits / 8);
    let mut ssnd = vec![0u8; 8];
    ssnd.extend((0..frames as usize * block).map(|i| (i % 251) as u8));
    chunks.push(aiff_chunk(b"SSND", &ssnd));
    form(form_type, &chunks)
}

/// An AIFF carrying **both** an `ID3 ` chunk and the IFF text chunks — the layout that pins
/// the documented precedence. `tail` puts the `ID3 ` chunk *after* the `SSND` audio, the
/// layout a tagger that appends produces and the one that costs a second positioned read.
pub fn aiff_id3(
    rate: u32,
    channels: u16,
    frames: u32,
    text: &[(&str, &str)],
    id3: &[(&str, &str)],
    tail: bool,
) -> Vec<u8> {
    let mut chunks = vec![comm_chunk(rate, channels, 16, frames, None)];
    for (id, value) in text {
        chunks.push(aiff_chunk(&chunk_id(id), value.as_bytes()));
    }
    let tag = aiff_chunk(b"ID3 ", &id3v2(3, id3, None));
    if !tail {
        chunks.push(tag.clone());
    }
    let mut ssnd = vec![0u8; 8];
    ssnd.extend((0..frames as usize * usize::from(channels) * 2).map(|i| (i % 251) as u8));
    chunks.push(aiff_chunk(b"SSND", &ssnd));
    if tail {
        chunks.push(tag);
    }
    form(b"AIFF", &chunks)
}

// --- FLAC (RFC 9639) --------------------------------------------------------------------

/// One metadata block: `[last:1|type:7]` + 24-bit big-endian length + body (§8.1).
pub fn flac_block(last: bool, kind: u8, body: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.push(if last { 0x80 | kind } else { kind });
    b.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
    b.extend_from_slice(body);
    b
}

/// A STREAMINFO body (§8.2): block sizes, frame sizes, then the packed
/// rate/channels/depth/sample-count field and the MD5.
pub fn streaminfo(rate: u32, channels: u32, bits: u32, total: u64) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&4096u16.to_be_bytes()); // minimum block size
    b.extend_from_slice(&4096u16.to_be_bytes()); // maximum block size
    b.extend_from_slice(&[0, 0, 0, 0, 0, 0]); // minimum/maximum frame size: unknown
    let packed = (u64::from(rate) << 44)
        | (u64::from(channels - 1) << 41)
        | (u64::from(bits - 1) << 36)
        | (total & 0x0000_000F_FFFF_FFFF);
    b.extend_from_slice(&packed.to_be_bytes());
    b.extend_from_slice(&[0u8; 16]); // MD5 of the unencoded audio
    b
}

/// A `VORBIS_COMMENT` body (§8.6): a length-prefixed vendor string, a count, then that many
/// length-prefixed `FIELD=value` entries — all lengths little-endian.
///
/// The same body serves Ogg Vorbis (Vorbis I §5.2.2.1) and Ogg Opus (RFC 7845 §5.2) behind
/// their own magics, which is why the Ogg builders below reuse it.
pub fn vorbis_comment<S: AsRef<str>>(entries: &[S]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(b"ref");
    b.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for e in entries {
        let e = e.as_ref();
        b.extend_from_slice(&(e.len() as u32).to_le_bytes());
        b.extend_from_slice(e.as_bytes());
    }
    b
}

/// A `PICTURE` body (§8.7): picture type, MIME, description, dimensions, then the image.
pub fn picture_block(mime: &str, data: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&3u32.to_be_bytes()); // 3 = cover (front)
    b.extend_from_slice(&(mime.len() as u32).to_be_bytes());
    b.extend_from_slice(mime.as_bytes());
    b.extend_from_slice(&0u32.to_be_bytes()); // empty description
    b.extend_from_slice(&[0u8; 16]); // width, height, colour depth, indexed colours
    b.extend_from_slice(&(data.len() as u32).to_be_bytes());
    b.extend_from_slice(data);
    b
}

/// A complete FLAC *metadata* stream: the `fLaC` marker (§8), STREAMINFO, an optional
/// `VORBIS_COMMENT` and an optional `PICTURE`, then a token frame-shaped tail so the file is
/// not pure metadata. Not decodable audio — nothing in this crate decodes.
pub fn flac(
    rate: u32,
    channels: u32,
    bits: u32,
    total: u64,
    comments: &[&str],
    picture: Option<(&str, &[u8])>,
) -> Vec<u8> {
    let mut f = b"fLaC".to_vec();
    let last_is_streaminfo = comments.is_empty() && picture.is_none();
    f.extend_from_slice(&flac_block(last_is_streaminfo, 0, &streaminfo(rate, channels, bits, total)));
    if !comments.is_empty() {
        f.extend_from_slice(&flac_block(picture.is_none(), 4, &vorbis_comment(comments)));
    }
    if let Some((mime, data)) = picture {
        f.extend_from_slice(&flac_block(true, 6, &picture_block(mime, data)));
    }
    // A frame sync (§9.1.1) and some bytes: enough that the file has a body.
    f.extend_from_slice(&[0xFF, 0xF8, 0x69, 0x18]);
    f.extend_from_slice(&[0u8; 60]);
    f
}

// --- MP3: ID3v2 / ID3v1 / APEv2 side-cars and MPEG audio frames ---------------------------

/// One ID3v2 frame. v2.3 (`id3v2.3.0` §3.3) writes a plain 32-bit big-endian size; v2.4
/// (`id3v2.4.0-structure` §4) writes a **synchsafe** one — reading either as the other is the
/// classic ID3 bug, so the builder distinguishes them the same way the parser does.
pub fn id3_frame(major: u8, id: &str, body: &[u8]) -> Vec<u8> {
    let mut f = id.as_bytes()[..4].to_vec();
    let size = body.len() as u32;
    if major >= 4 {
        f.extend_from_slice(&synchsafe(size));
    } else {
        f.extend_from_slice(&size.to_be_bytes());
    }
    f.extend_from_slice(&[0, 0]); // frame flags
    f.extend_from_slice(body);
    f
}

/// A 28-bit synchsafe integer (`id3v2.4.0-structure` §6.2): seven significant bits per octet,
/// high bit always zero, so a size field can never contain a false frame sync.
fn synchsafe(n: u32) -> [u8; 4] {
    [
        ((n >> 21) & 0x7F) as u8,
        ((n >> 14) & 0x7F) as u8,
        ((n >> 7) & 0x7F) as u8,
        (n & 0x7F) as u8,
    ]
}

/// A complete ID3v2 tag: the 10-byte header (§3.1) then the frames.
///
/// `frames` are `(frame id, value)`. A `TXXX` entry's value is `description=value` (§4.2.6 —
/// the description *is* the key, which is how `REPLAYGAIN_TRACK_GAIN` is written); every
/// other id is a text-information frame (§4.2). Text is ISO-8859-1 (encoding `$00`), which is
/// legal in every ID3v2 version and exact for the ASCII these fixtures use. `picture` adds an
/// `APIC` frame (§4.14).
pub fn id3v2(major: u8, frames: &[(&str, &str)], picture: Option<(&str, &[u8])>) -> Vec<u8> {
    let mut body = Vec::new();
    for (id, value) in frames {
        let mut b = vec![0u8]; // text encoding $00 — ISO-8859-1
        if *id == "TXXX" {
            let (desc, v) = value.split_once('=').unwrap_or(("COMMENT", value));
            b.extend_from_slice(desc.as_bytes());
            b.push(0); // the description's terminator
            b.extend_from_slice(v.as_bytes());
        } else {
            b.extend_from_slice(value.as_bytes());
        }
        body.extend_from_slice(&id3_frame(major, id, &b));
    }
    if let Some((mime, data)) = picture {
        let mut b = vec![0u8]; // text encoding, for the description
        b.extend_from_slice(mime.as_bytes());
        b.push(0);
        b.push(3); // picture type 3 = cover (front)
        b.push(0); // empty description
        b.extend_from_slice(data);
        body.extend_from_slice(&id3_frame(major, "APIC", &b));
    }
    let mut tag = b"ID3".to_vec();
    tag.push(major);
    tag.push(0); // revision
    tag.push(0); // flags: no unsynchronisation, no extended header, no footer
    tag.extend_from_slice(&synchsafe(body.len() as u32));
    tag.extend_from_slice(&body);
    tag
}

/// A trailing ID3v1 tag: the flat 128-byte record of Eric Kemp's 1996 note, with Michael
/// Mutschler's v1.1 track amendment (`comment[28] == $00`, `comment[29] == track`).
pub fn id3v1(title: &str, artist: &str, album: &str, year: &str, track: u8) -> Vec<u8> {
    let mut t = vec![0u8; 128];
    t[..3].copy_from_slice(b"TAG");
    let put = |dst: &mut [u8], s: &str| {
        let n = s.len().min(dst.len());
        dst[..n].copy_from_slice(&s.as_bytes()[..n]);
    };
    put(&mut t[3..33], title);
    put(&mut t[33..63], artist);
    put(&mut t[63..93], album);
    put(&mut t[93..97], year);
    t[126] = track; // comment[29] — the v1.1 track byte; comment[28] stays $00
    t[127] = 0xFF; // genre index $FF: unset
    t
}

/// An APEv2 tag with a footer and no header (`APEv2_specification`): the items, then the
/// 32-octet footer whose `size` field covers items **plus** footer.
pub fn ape_tag(items: &[(&str, &str)]) -> Vec<u8> {
    let mut t = Vec::new();
    for (key, value) in items {
        t.extend_from_slice(&(value.len() as u32).to_le_bytes());
        t.extend_from_slice(&0u32.to_le_bytes()); // item flags: UTF-8 text
        t.extend_from_slice(key.as_bytes());
        t.push(0); // the key's NUL terminator
        t.extend_from_slice(value.as_bytes());
    }
    let size = (t.len() + 32) as u32;
    t.extend_from_slice(b"APETAGEX");
    t.extend_from_slice(&2000u32.to_le_bytes()); // APEv2
    t.extend_from_slice(&size.to_le_bytes());
    t.extend_from_slice(&(items.len() as u32).to_le_bytes());
    t.extend_from_slice(&0u32.to_le_bytes()); // flags: this is a footer, and there is no header
    t.extend_from_slice(&[0u8; 8]); // reserved
    t
}

/// An MPEG-1 Layer III frame header (ISO/IEC 11172-3 §2.4.1.3): the 11-bit sync word, version
/// `11` (MPEG-1), layer `01` (Layer III), no CRC, then the bitrate and sampling-frequency
/// indices of §2.4.2.3 and channel mode `00` (stereo).
fn mp3_header(rate: u32, kbps: u32) -> [u8; 4] {
    const BITRATES: [u32; 15] = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
    let index = BITRATES.iter().position(|&b| b == kbps).unwrap_or(9) as u8;
    let freq = match rate {
        48_000 => 1u8,
        32_000 => 2,
        _ => 0, // 44100
    };
    [0xFF, 0xFB, (index << 4) | (freq << 2), 0x00]
}

/// Bytes in one MPEG-1 Layer III frame: `144 × bitrate / sampling_frequency` (§2.4.3.1), with
/// no padding bit set.
fn mp3_frame_len(rate: u32, kbps: u32) -> usize {
    144 * kbps as usize * 1000 / rate as usize
}

/// `count` constant-bitrate MPEG-1 Layer III frames. The payload is zeros — nothing in this
/// crate decodes, and a zero payload cannot fake a frame sync inside a frame.
pub fn mp3_frames(rate: u32, kbps: u32, count: usize) -> Vec<u8> {
    let header = mp3_header(rate, kbps);
    let len = mp3_frame_len(rate, kbps);
    let mut out = Vec::new();
    for _ in 0..count {
        let mut frame = vec![0u8; len];
        frame[..4].copy_from_slice(&header);
        out.extend_from_slice(&frame);
    }
    out
}

/// A Xing header frame: a normal frame whose (unused) payload carries the VBR summary the
/// Xing SDK defined and LAME writes. For MPEG-1 stereo without a CRC word the tag begins at
/// `4 + 32` — the frame header plus the Layer III side-information block (§2.4.1.7).
pub fn mp3_xing_frame(rate: u32, kbps: u32, frames: u32) -> Vec<u8> {
    let mut frame = vec![0u8; mp3_frame_len(rate, kbps)];
    frame[..4].copy_from_slice(&mp3_header(rate, kbps));
    frame[36..40].copy_from_slice(b"Xing");
    frame[40..44].copy_from_slice(&1u32.to_be_bytes()); // flags: a frame count follows
    frame[44..48].copy_from_slice(&frames.to_be_bytes());
    frame
}

/// A complete MP3: `[ID3v2] [Xing frame] frames… [APEv2] [ID3v1]`, the layout `mp3.rs`
/// documents. `frames` counts the audio frames and is what a Xing header declares.
#[allow(clippy::too_many_arguments)] // a fixture builder: one parameter per optional side-car
pub fn mp3(
    rate: u32,
    kbps: u32,
    frames: usize,
    v2: &[(&str, &str)],
    picture: Option<(&str, &[u8])>,
    xing: bool,
    v1: Option<(&str, &str, &str, &str, u8)>,
    ape: &[(&str, &str)],
) -> Vec<u8> {
    let mut f = Vec::new();
    if !v2.is_empty() || picture.is_some() {
        f.extend_from_slice(&id3v2(3, v2, picture));
    }
    if xing {
        f.extend_from_slice(&mp3_xing_frame(rate, kbps, frames as u32));
    }
    f.extend_from_slice(&mp3_frames(rate, kbps, frames));
    if !ape.is_empty() {
        f.extend_from_slice(&ape_tag(ape));
    }
    if let Some((t, a, al, y, n)) = v1 {
        f.extend_from_slice(&id3v1(t, a, al, y, n));
    }
    f
}

// --- Musepack (Musepack SV8/SV7 specifications; libmpc reference reader) ------------------

/// A plain Musepack varint: big-endian base-128, seven payload bits per octet in the low
/// bits, the MSB set on every octet but the last.
pub fn mpc_varint(value: u64) -> Vec<u8> {
    let mut groups = Vec::new();
    let mut v = value;
    loop {
        groups.push((v & 0x7F) as u8);
        v >>= 7;
        if v == 0 {
            break;
        }
    }
    groups.reverse();
    let last = groups.len() - 1;
    for (i, g) in groups.iter_mut().enumerate() {
        if i != last {
            *g |= 0x80;
        }
    }
    groups
}

/// One SV8 packet: `<2 ASCII key><varint size><payload>`, where the size **includes the key
/// and the size octets themselves**. The width of the size field therefore depends on the
/// value it is about to hold, so it is solved for rather than computed.
pub fn mpc_packet(key: &[u8; 2], payload: &[u8]) -> Vec<u8> {
    let mut width = 1;
    loop {
        let encoded = mpc_varint((payload.len() + 2 + width) as u64);
        if encoded.len() == width {
            let mut v = key.to_vec();
            v.extend_from_slice(&encoded);
            v.extend_from_slice(payload);
            return v;
        }
        width += 1;
    }
}

/// A complete SV8 file: `MPCK`, any `extra` packets, the `SH` stream header, then the `RG`,
/// `AP` and `SE` packets a real file carries.
///
/// `extra` packets are written **before** `SH` so a test can prove the walk steps over
/// packets it does not recognise; a real encoder writes `SH` first.
pub fn mpc_sv8(
    rate_index: u8,
    channels: u8,
    samples: u64,
    silence: u64,
    extra: &[([u8; 2], Vec<u8>)],
) -> Vec<u8> {
    let mut sh = vec![0u8; 4]; // CRC32 — a tag scan does not verify it
    sh.push(8); // stream version
    sh.extend_from_slice(&mpc_varint(samples));
    sh.extend_from_slice(&mpc_varint(silence));
    // freq index (3 bits) | max used bands - 1 (5 bits): 28 bands is a typical value.
    sh.push((rate_index << 5) | 27);
    // channels - 1 (4 bits) | mid/side (1) | audio block frames (3): 4^3 = 64 frames.
    sh.push((channels.saturating_sub(1) << 4) | (1 << 3) | 3);

    let mut f = b"MPCK".to_vec();
    for (key, payload) in extra {
        f.extend_from_slice(&mpc_packet(key, payload));
    }
    f.extend_from_slice(&mpc_packet(b"SH", &sh));
    f.extend_from_slice(&mpc_packet(b"RG", &[1, 0, 0, 0, 0, 0, 0, 0, 0]));
    f.extend_from_slice(&mpc_packet(b"AP", &[0x5Au8; 256]));
    f.extend_from_slice(&mpc_packet(b"SE", &[]));
    f
}

/// A complete SV7 file: `MP+`, the version octet, and the 28-octet header of little-endian
/// words whose fields are bit-packed MSB-first, then a token body.
pub fn mpc_sv7(rate_index: u8, frames: u32, true_gapless: bool, last_frame_samples: u16) -> Vec<u8> {
    let mut f = b"MP+".to_vec();
    f.push(0x07); // low nibble 7 = stream version; high nibble 0 = PNS unused
    f.extend_from_slice(&frames.to_le_bytes()); // W1: FrameCount
    // W2: bit 31 intensity stereo, 30 mid/side, 24..29 MaxBand, 20..23 Profile,
    //     18..19 Link, 16..17 sample frequency, 0..15 MaxLevel.
    let w2 = (1u32 << 30) | (28u32 << 24) | (10u32 << 20) | ((u32::from(rate_index) & 3) << 16) | 0x7FFF;
    f.extend_from_slice(&w2.to_le_bytes());
    f.extend_from_slice(&0u32.to_le_bytes()); // W3: title gain / peak
    f.extend_from_slice(&0u32.to_le_bytes()); // W4: album gain / peak
    // W5: bit 31 TrueGapless, bits 20..30 LastFrameLength, bit 19 fast-seek, rest unused.
    let w5 = (u32::from(true_gapless) << 31) | ((u32::from(last_frame_samples) & 0x7FF) << 20);
    f.extend_from_slice(&w5.to_le_bytes());
    f.extend_from_slice(&(106u32 << 24).to_le_bytes()); // W6: encoder version 1.06
    f.extend((0..512).map(|i| (i % 251) as u8));
    f
}

// --- Monkey's Audio (Monkey's Audio SDK: APE_DESCRIPTOR / APE_HEADER / APE_HEADER_OLD) ----

/// A complete modern (`nVersion >= 3980`) Monkey's Audio file: `APE_DESCRIPTOR`,
/// `APE_HEADER`, and a token frame body so the file is not pure header.
///
/// The compression level is written as 2000 (`normal`); a test that needs another patches it
/// at `52 + 0`.
pub fn ape_file(
    version: u16,
    rate: u32,
    channels: u16,
    blocks_per_frame: u32,
    total_frames: u32,
    final_frame_blocks: u32,
) -> Vec<u8> {
    // --- APE_DESCRIPTOR, 52 octets ---
    let mut d = b"MAC ".to_vec();
    d.extend_from_slice(&version.to_le_bytes());
    d.extend_from_slice(&0u16.to_le_bytes()); // nPadding — real octets, meaningless value
    d.extend_from_slice(&52u32.to_le_bytes()); // nDescriptorBytes
    d.extend_from_slice(&24u32.to_le_bytes()); // nHeaderBytes
    d.extend_from_slice(&0u32.to_le_bytes()); // nSeekTableBytes
    d.extend_from_slice(&0u32.to_le_bytes()); // nHeaderDataBytes
    d.extend_from_slice(&512u32.to_le_bytes()); // nAPEFrameDataBytes
    d.extend_from_slice(&0u32.to_le_bytes()); // nAPEFrameDataBytesHigh
    d.extend_from_slice(&0u32.to_le_bytes()); // nTerminatingDataBytes
    d.extend_from_slice(&[0u8; 16]); // cFileMD5

    // --- APE_HEADER, 24 octets ---
    d.extend_from_slice(&2000u16.to_le_bytes()); // nCompressionLevel: normal
    d.extend_from_slice(&0u16.to_le_bytes()); // nFormatFlags
    d.extend_from_slice(&blocks_per_frame.to_le_bytes());
    d.extend_from_slice(&final_frame_blocks.to_le_bytes());
    d.extend_from_slice(&total_frames.to_le_bytes());
    d.extend_from_slice(&16u16.to_le_bytes()); // nBitsPerSample
    d.extend_from_slice(&channels.to_le_bytes());
    d.extend_from_slice(&rate.to_le_bytes());

    d.extend((0..512).map(|i| (i % 251) as u8)); // compressed frame data, never read
    d
}

/// A complete pre-3.98 (`nVersion < 3980`) Monkey's Audio file: the single 32-octet
/// `APE_HEADER_OLD`, then whatever its format flags promise — a peak level, a seek-element
/// count, a seek table — and a token body.
///
/// `nBlocksPerFrame` and `nBitsPerSample` are **not** written: an old file derives them from
/// the version and the format flags, which is exactly what the parser must reproduce.
pub fn ape_file_old(
    version: u16,
    level: u16,
    flags: u16,
    rate: u32,
    channels: u16,
    total_frames: u32,
    final_frame_blocks: u32,
) -> Vec<u8> {
    let mut h = b"MAC ".to_vec();
    h.extend_from_slice(&version.to_le_bytes());
    h.extend_from_slice(&level.to_le_bytes());
    h.extend_from_slice(&flags.to_le_bytes());
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&0u32.to_le_bytes()); // nHeaderBytes
    h.extend_from_slice(&0u32.to_le_bytes()); // nTerminatingBytes
    h.extend_from_slice(&total_frames.to_le_bytes());
    h.extend_from_slice(&final_frame_blocks.to_le_bytes());

    // The conditional fields, in the order the SDK reads them.
    if flags & 4 != 0 {
        h.extend_from_slice(&0u32.to_le_bytes()); // nPeakLevel
    }
    let seek_elements = if flags & 16 != 0 {
        h.extend_from_slice(&total_frames.to_le_bytes()); // nSeekTableElements
        total_frames
    } else {
        total_frames
    };
    if flags & 32 == 0 {
        // CREATE_WAV_HEADER clear means a stored WAV header follows — of nHeaderBytes, which
        // this builder writes as zero.
    }
    h.extend(std::iter::repeat_n(0u8, seek_elements as usize * 4)); // the seek byte table
    if version <= 3800 {
        h.extend(std::iter::repeat_n(0u8, seek_elements as usize)); // the seek bit table
    }
    h.extend((0..512).map(|i| (i % 251) as u8));
    h
}

// --- WavPack (WavPack 4 & 5 Binary File / Block Format, David Bryant) ---------------------

/// One metadata sub-block (§3.0): the id octet, the payload size in 16-bit **words**, then
/// the payload padded to an even number of bytes. `0x40` is set in the id when the payload's
/// real length is odd, and `0x80` when the size needs three octets rather than one.
pub fn wv_sub_block(id: u8, data: &[u8]) -> Vec<u8> {
    let words = data.len().div_ceil(2);
    let mut v = Vec::new();
    let odd = if data.len() % 2 == 1 { 0x40u8 } else { 0 };
    if words > 255 {
        v.push(id | odd | 0x80);
        v.extend_from_slice(&(words as u32).to_le_bytes()[..3]);
    } else {
        v.push(id | odd);
        v.push(words as u8);
    }
    v.extend_from_slice(data);
    if data.len() % 2 == 1 {
        v.push(0);
    }
    v
}

/// One WavPack block: the 32-byte little-endian header of §2.0 followed by its metadata
/// sub-blocks. `ckSize` is "size of entire block (minus 8)".
pub fn wavpack_block(
    version: u16,
    total_samples: u32,
    block_index: u32,
    block_samples: u32,
    flags: u32,
    subs: &[Vec<u8>],
) -> Vec<u8> {
    let body: Vec<u8> = subs.concat();
    let mut v = b"wvpk".to_vec();
    v.extend_from_slice(&((24 + body.len()) as u32).to_le_bytes()); // ckSize
    v.extend_from_slice(&version.to_le_bytes());
    v.push(0); // block_index_u8   — upper 8 bits of the 40-bit index
    v.push(0); // total_samples_u8 — upper 8 bits of the 40-bit count
    v.extend_from_slice(&total_samples.to_le_bytes());
    v.extend_from_slice(&block_index.to_le_bytes());
    v.extend_from_slice(&block_samples.to_le_bytes());
    v.extend_from_slice(&flags.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes()); // crc of the decoded data
    v.extend_from_slice(&body);
    v
}

/// A complete single-block WavPack file of `total_samples` frames.
///
/// `rate` **0** selects flags rate index 15 — "unknown/custom" (§2.0) — which is how a file
/// that carries its rate in an `ID_SAMPLE_RATE` sub-block is built. Any other rate is looked
/// up in the standard table; a rate that is not in it also becomes index 15.
///
/// The table is written out here rather than shared with `wv.rs`, deliberately: a fixture
/// that imports the parser's own constant cannot catch a wrong constant.
pub fn wavpack(rate: u32, channels: u16, total_samples: u32, subs: &[Vec<u8>]) -> Vec<u8> {
    const RATES: [u32; 15] = [
        6_000, 8_000, 9_600, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
        64_000, 88_200, 96_000, 192_000,
    ];
    let index = RATES.iter().position(|&r| r == rate).unwrap_or(15) as u32;
    // bits 1,0 = 01 (2 bytes/sample); bit 2 = mono; bits 26-23 = the rate index;
    // bits 11,12 = initial/final block in sequence, which a mono or stereo file always sets.
    let flags = 0x01
        | if channels == 1 { 1 << 2 } else { 0 }
        | (index << 23)
        | (1 << 11)
        | (1 << 12);
    wavpack_block(0x410, total_samples, 0, total_samples, flags, subs)
}

// --- MP4 / M4A (ISO/IEC 14496-12 + QTFF Metadata) ----------------------------------------

/// One ISO-BMFF box: `<u32 size><type><body>` (§4.2).
pub fn mp4_box(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut b = ((body.len() + 8) as u32).to_be_bytes().to_vec();
    b.extend_from_slice(kind);
    b.extend_from_slice(body);
    b
}

/// A four-character code from a `&str`, mapping a leading `©` to the single Mac OS Roman
/// octet `0xA9` Apple's QuickTime keys use — *not* the two-octet UTF-8 encoding of U+00A9,
/// which would make the four-CC five octets long (QTFF Metadata).
fn fourcc(key: &str) -> [u8; 4] {
    let mut out = [b' '; 4];
    for (i, c) in key.chars().take(4).enumerate() {
        out[i] = if c == '©' { 0xA9 } else { c as u8 };
    }
    out
}

/// One metadata item `data` atom (QTFF Metadata): a 4-octet type indicator (type set 0 = the
/// well-known types, then the 3-octet type number) and a 4-octet locale, then the value.
fn ilst_data(well_known: u32, value: &[u8]) -> Vec<u8> {
    let mut b = well_known.to_be_bytes().to_vec();
    b.extend_from_slice(&0u32.to_be_bytes()); // locale
    b.extend_from_slice(value);
    mp4_box(b"data", &b)
}

/// A complete minimal `.m4a`: `ftyp`, `moov` (a sound `trak` plus the iTunes `udta/meta/ilst`)
/// and `mdat`, in that order when `faststart`, otherwise `ftyp mdat moov` — the layout most
/// encoders write and the one that costs the scanner a second read.
///
/// `text` are `(item id, value)` pairs (`©nam`, `©ART`, `aART`, …); `track` becomes a `trkn`
/// pair and `cover` a `covr` image.
#[allow(clippy::too_many_arguments)] // a fixture builder: one parameter per optional box
pub fn m4a(
    rate: u32,
    channels: u16,
    samples: u64,
    text: &[(&str, &str)],
    track: Option<u16>,
    cover: Option<(&str, &[u8])>,
    faststart: bool,
) -> Vec<u8> {
    // --- ftyp (§4.3) ---
    let mut ftyp = b"M4A ".to_vec();
    ftyp.extend_from_slice(&0u32.to_be_bytes()); // minor version
    ftyp.extend_from_slice(b"M4A mp42isom");
    let ftyp = mp4_box(b"ftyp", &ftyp);

    // --- moov/mvhd (§8.2.2), version 0: 32-bit times, timescale 1000 ---
    let movie_ticks = (samples * 1000 / u64::from(rate)) as u32;
    let mut mvhd = vec![0u8; 4]; // version 0 + flags
    mvhd.extend_from_slice(&[0u8; 8]); // creation / modification time
    mvhd.extend_from_slice(&1000u32.to_be_bytes()); // timescale
    mvhd.extend_from_slice(&movie_ticks.to_be_bytes());
    mvhd.extend_from_slice(&0x0001_0000u32.to_be_bytes()); // rate 1.0
    mvhd.extend_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
    mvhd.extend_from_slice(&[0u8; 10]); // reserved
    mvhd.extend_from_slice(&[0u8; 36]); // transform matrix
    mvhd.extend_from_slice(&[0u8; 24]); // pre_defined
    mvhd.extend_from_slice(&2u32.to_be_bytes()); // next_track_ID
    let mvhd = mp4_box(b"mvhd", &mvhd);

    // --- trak/mdia/mdhd (§8.4.2): the media timeline, at the *media* timescale ---
    let mut mdhd = vec![0u8; 4];
    mdhd.extend_from_slice(&[0u8; 8]);
    mdhd.extend_from_slice(&rate.to_be_bytes()); // timescale == the sample rate
    mdhd.extend_from_slice(&(samples as u32).to_be_bytes());
    mdhd.extend_from_slice(&0x55C4u16.to_be_bytes()); // language "und"
    mdhd.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    let mdhd = mp4_box(b"mdhd", &mdhd);

    // --- hdlr (§8.4.3): version/flags, pre_defined, then the handler four-CC ---
    let mut hdlr = vec![0u8; 4];
    hdlr.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
    hdlr.extend_from_slice(b"soun");
    hdlr.extend_from_slice(&[0u8; 12]); // reserved
    hdlr.push(0); // empty name
    let hdlr = mp4_box(b"hdlr", &hdlr);

    // --- stsd (§8.5.2) with one `mp4a` AudioSampleEntry (§12.2.3) ---
    let mut mp4a = vec![0u8; 6]; // SampleEntry reserved
    mp4a.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
    mp4a.extend_from_slice(&[0u8; 8]); // version, revision, vendor
    mp4a.extend_from_slice(&channels.to_be_bytes());
    mp4a.extend_from_slice(&16u16.to_be_bytes()); // samplesize
    mp4a.extend_from_slice(&[0u8; 4]); // pre_defined + reserved
    mp4a.extend_from_slice(&(rate << 16).to_be_bytes()); // 16.16 fixed-point sample rate
    // A token `esds` (§14 / ISO/IEC 14496-1): present because every real `mp4a` has one; the
    // scanner reads the entry prologue above and never looks inside it.
    mp4a.extend_from_slice(&mp4_box(b"esds", &[0u8; 4]));
    let mp4a = mp4_box(b"mp4a", &mp4a);

    let mut stsd = vec![0u8; 4]; // version + flags
    stsd.extend_from_slice(&1u32.to_be_bytes()); // entry_count
    stsd.extend_from_slice(&mp4a);
    let stsd = mp4_box(b"stsd", &stsd);

    // Empty sample tables: a scanner never resolves samples, but a well-formed `stbl` has them.
    let empty = |kind: &[u8; 4]| {
        let mut b = vec![0u8; 4];
        b.extend_from_slice(&0u32.to_be_bytes());
        mp4_box(kind, &b)
    };
    let stbl = mp4_box(
        b"stbl",
        &[stsd, empty(b"stts"), empty(b"stsc"), empty(b"stco"), {
            let mut b = vec![0u8; 4];
            b.extend_from_slice(&0u32.to_be_bytes()); // sample_size
            b.extend_from_slice(&0u32.to_be_bytes()); // sample_count
            mp4_box(b"stsz", &b)
        }]
        .concat(),
    );
    let smhd = mp4_box(b"smhd", &[0u8; 8]);
    let minf = mp4_box(b"minf", &[smhd, stbl].concat());
    let mdia = mp4_box(b"mdia", &[mdhd, hdlr, minf].concat());

    let mut tkhd = vec![0u8, 0, 0, 7]; // version 0, flags: enabled | in movie | in preview
    tkhd.extend_from_slice(&[0u8; 8]); // times
    tkhd.extend_from_slice(&1u32.to_be_bytes()); // track_ID
    tkhd.extend_from_slice(&0u32.to_be_bytes()); // reserved
    tkhd.extend_from_slice(&movie_ticks.to_be_bytes());
    tkhd.extend_from_slice(&[0u8; 52]); // reserved, layer, volume, matrix
    tkhd.extend_from_slice(&[0u8; 8]); // width, height (0 for audio)
    let trak = mp4_box(b"trak", &[mp4_box(b"tkhd", &tkhd), mdia].concat());

    // --- udta/meta/ilst (§8.10.1, §8.11.1, QTFF Metadata) ---
    let mut ilst = Vec::new();
    for (id, value) in text {
        ilst.extend_from_slice(&mp4_box(&fourcc(id), &ilst_data(1, value.as_bytes())));
    }
    if let Some(n) = track {
        // `trkn` is implicit-typed: 2 reserved octets, the 1-based index, the total, padding.
        let mut b = vec![0u8, 0];
        b.extend_from_slice(&n.to_be_bytes());
        b.extend_from_slice(&0u16.to_be_bytes());
        b.extend_from_slice(&[0u8, 0]);
        ilst.extend_from_slice(&mp4_box(b"trkn", &ilst_data(0, &b)));
    }
    if let Some((mime, data)) = cover {
        let ty = if mime == "image/png" { 14 } else { 13 };
        ilst.extend_from_slice(&mp4_box(b"covr", &ilst_data(ty, data)));
    }
    let mut meta = vec![0u8; 4]; // `meta` is a FullBox: version + flags precede its children
    meta.extend_from_slice(&mp4_box(b"hdlr", &{
        let mut h = vec![0u8; 4];
        h.extend_from_slice(&0u32.to_be_bytes());
        h.extend_from_slice(b"mdir");
        h.extend_from_slice(b"appl");
        h.extend_from_slice(&[0u8; 9]);
        h
    }));
    meta.extend_from_slice(&mp4_box(b"ilst", &ilst));
    let udta = mp4_box(b"udta", &[mp4_box(b"meta", &meta)].concat());

    let moov = mp4_box(b"moov", &[mvhd, trak, udta].concat());
    let mdat = mp4_box(b"mdat", &vec![0x4Du8; 4096]);

    if faststart {
        [ftyp, moov, mdat].concat()
    } else {
        [ftyp, mdat, moov].concat()
    }
}

// --- Ogg (RFC 3533) ----------------------------------------------------------------------

/// Lacing values for one packet (RFC 3533 §6, field 9): `len / 255` values of 255 then one
/// final value below 255, which is what terminates the packet. A length that is an exact
/// multiple of 255 therefore ends with a lacing value of 0 (§5).
fn lacing(len: usize) -> Vec<u8> {
    let mut v = vec![255u8; len / 255];
    v.push((len % 255) as u8);
    v
}

/// Write one packet as however many pages it needs (at most 255 lacing values per page, §6).
/// Intermediate pages carry the `-1` granule sentinel — "no packets finish on this page".
fn ogg_packet(out: &mut Vec<u8>, serial: u32, seq: &mut u32, packet: &[u8], flags: u8, granule: u64) {
    use pf_ogg::page::{write_page, GRANULE_NONE};
    let table = lacing(packet.len());
    let pages = table.len().div_ceil(255);
    let mut at = 0usize;
    for (i, chunk) in table.chunks(255).enumerate() {
        let bytes: usize = chunk.iter().map(|&b| b as usize).sum();
        let mut header = if i == 0 { flags & pf_ogg::page::flags::BOS } else { pf_ogg::page::flags::CONTINUED };
        if i + 1 == pages {
            header |= flags & pf_ogg::page::flags::EOS;
        }
        let g = if i + 1 == pages { granule } else { GRANULE_NONE };
        write_page(out, header, g, serial, *seq, chunk, &packet[at..at + bytes]);
        *seq += 1;
        at += bytes;
    }
}

/// Base64 (RFC 4648 §4) — the encoding a `METADATA_BLOCK_PICTURE` comment value uses,
/// because a Vorbis comment value is text and a picture block is not.
fn base64(data: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { A[n as usize & 63] as char } else { '=' });
    }
    out
}

/// The comment entries plus, when a picture is given, the `METADATA_BLOCK_PICTURE` entry that
/// carries it: the FLAC `PICTURE` block body (RFC 9639 §8.8) base64-encoded into the value.
fn comment_entries(comments: &[&str], picture: Option<(&str, &[u8])>) -> Vec<String> {
    let mut v: Vec<String> = comments.iter().map(|s| (*s).to_string()).collect();
    if let Some((mime, data)) = picture {
        v.push(format!("METADATA_BLOCK_PICTURE={}", base64(&picture_block(mime, data))));
    }
    v
}

/// A complete Ogg Opus stream: the `OpusHead` bos page (RFC 7845 §5.1), the `OpusTags` page
/// (§5.2), and one audio page whose granule position is `last_granule` — the 48 kHz sample
/// count that, minus `pre_skip`, is the duration (§4).
pub fn ogg_opus(
    channels: u8,
    pre_skip: u16,
    input_rate: u32,
    comments: &[&str],
    picture: Option<(&str, &[u8])>,
    last_granule: u64,
) -> Vec<u8> {
    let mut head = b"OpusHead".to_vec();
    head.push(1); // version
    head.push(channels);
    head.extend_from_slice(&pre_skip.to_le_bytes());
    head.extend_from_slice(&input_rate.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes()); // output gain
    head.push(0); // channel mapping family 0

    let mut tags = b"OpusTags".to_vec();
    tags.extend_from_slice(&vorbis_comment(&comment_entries(comments, picture)));

    let mut f = Vec::new();
    let mut seq = 0u32;
    ogg_packet(&mut f, 0x5C0F_FEE5, &mut seq, &head, pf_ogg::page::flags::BOS, 0);
    ogg_packet(&mut f, 0x5C0F_FEE5, &mut seq, &tags, 0, 0);
    ogg_packet(&mut f, 0x5C0F_FEE5, &mut seq, &[0xFCu8; 160], pf_ogg::page::flags::EOS, last_granule);
    f
}

/// A complete Ogg Vorbis stream: the identification, comment and (token) setup header packets
/// (Vorbis I §4.2), then one audio page carrying `last_granule` PCM samples.
pub fn ogg_vorbis(channels: u8, rate: u32, comments: &[&str], last_granule: u64) -> Vec<u8> {
    let mut ident = b"\x01vorbis".to_vec();
    ident.extend_from_slice(&0u32.to_le_bytes()); // vorbis_version
    ident.push(channels);
    ident.extend_from_slice(&rate.to_le_bytes());
    ident.extend_from_slice(&0u32.to_le_bytes()); // bitrate maximum
    ident.extend_from_slice(&192_000u32.to_le_bytes()); // bitrate nominal
    ident.extend_from_slice(&0u32.to_le_bytes()); // bitrate minimum
    ident.push(0xB8); // blocksize_0 = 2^8, blocksize_1 = 2^11
    ident.push(0x01); // framing flag

    let mut comment = b"\x03vorbis".to_vec();
    comment.extend_from_slice(&vorbis_comment(comments));
    comment.push(0x01); // §5.2.2.1's framing bit

    let mut setup = b"\x05vorbis".to_vec();
    setup.extend_from_slice(&[0u8; 64]); // token codebooks: never read by a tag scan

    let mut f = Vec::new();
    let mut seq = 0u32;
    let serial = 0x0A11_CE00;
    ogg_packet(&mut f, serial, &mut seq, &ident, pf_ogg::page::flags::BOS, 0);
    ogg_packet(&mut f, serial, &mut seq, &comment, 0, 0);
    ogg_packet(&mut f, serial, &mut seq, &setup, 0, 0);
    ogg_packet(&mut f, serial, &mut seq, &[0x55u8; 96], pf_ogg::page::flags::EOS, last_granule);
    f
}

/// A complete Ogg Speex stream (Speex manual §7.3, Table 7.1): the 80-octet `"Speex   "`
/// header on the bos page, then the comment packet — which carries **no magic, no packet-type
/// octet and no framing bit**, just the bare Vorbis comment body — then one audio page whose
/// granule position is the sample count at `rate`.
pub fn ogg_speex(
    rate: u32,
    channels: u8,
    comments: &[&str],
    picture: Option<(&str, &[u8])>,
    last_granule: u64,
) -> Vec<u8> {
    let mut head = b"Speex   ".to_vec();
    let mut version = [0u8; 20];
    version[..11].copy_from_slice(b"speex-1.2rc");
    head.extend_from_slice(&version);
    // The thirteen little-endian 32-bit fields, in Table 7.1's order.
    for f in [
        1i32,                  // speex_version_id
        80,                    // header_size
        rate as i32,           // rate
        1,                     // mode: wideband
        4,                     // mode_bitstream_version
        i32::from(channels),   // nb_channels
        27_800,                // bitrate
        320,                   // frame_size
        0,                     // vbr
        1,                     // frames_per_packet
        0,                     // extra_headers
        0,                     // reserved1
        0,                     // reserved2
    ] {
        head.extend_from_slice(&f.to_le_bytes());
    }

    let comment = vorbis_comment(&comment_entries(comments, picture));

    let mut f = Vec::new();
    let mut seq = 0u32;
    let serial = 0x5BEE_0001;
    ogg_packet(&mut f, serial, &mut seq, &head, pf_ogg::page::flags::BOS, 0);
    ogg_packet(&mut f, serial, &mut seq, &comment, 0, 0);
    ogg_packet(&mut f, serial, &mut seq, &[0x3Cu8; 128], pf_ogg::page::flags::EOS, last_granule);
    f
}

/// A complete Ogg FLAC stream ("Ogg Mapping for FLAC" §3): packet 1 is the mapping header
/// followed by the native `fLaC` signature and STREAMINFO, and each remaining metadata block
/// is its own packet. The final page's granule position is the PCM sample count (§4).
pub fn ogg_flac(
    rate: u32,
    channels: u32,
    total_samples: u64,
    comments: &[&str],
    picture: Option<(&str, &[u8])>,
) -> Vec<u8> {
    let extra = 1 + u16::from(picture.is_some()); // header packets after the first
    let mut first = b"\x7fFLAC".to_vec();
    first.extend_from_slice(&[1, 0]); // mapping version 1.0
    first.extend_from_slice(&extra.to_be_bytes());
    first.extend_from_slice(b"fLaC");
    first.extend_from_slice(&flac_block(false, 0, &streaminfo(rate, channels, 16, total_samples)));

    let comment = flac_block(picture.is_none(), 4, &vorbis_comment(comments));

    let mut f = Vec::new();
    let mut seq = 0u32;
    let serial = 0xF1AC_0001;
    ogg_packet(&mut f, serial, &mut seq, &first, pf_ogg::page::flags::BOS, 0);
    ogg_packet(&mut f, serial, &mut seq, &comment, 0, 0);
    if let Some((mime, data)) = picture {
        let block = flac_block(true, 6, &picture_block(mime, data));
        ogg_packet(&mut f, serial, &mut seq, &block, 0, 0);
    }
    ogg_packet(&mut f, serial, &mut seq, &[0xFFu8, 0xF8, 0x69, 0x18], pf_ogg::page::flags::EOS, total_samples);
    f
}

// --- Matroska / WebM (RFC 9559; EBML per RFC 8794) ---------------------------------------

/// A master element: its pre-encoded ID, the shortest-valid size VINT, then the children
/// concatenated (RFC 8794 §5, §6.1). Built on `pf_mkv::ebml`'s own writers, so a fixture
/// exercises the same byte grammar the muxer emits.
pub fn mkv_master(element_id: &[u8], children: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = children.concat();
    let mut out = Vec::new();
    pf_mkv::ebml::write_id(&mut out, element_id);
    pf_mkv::ebml::write_size(&mut out, body.len() as u64);
    out.extend_from_slice(&body);
    out
}

fn mkv_uint(element_id: &[u8], v: u64) -> Vec<u8> {
    let mut out = Vec::new();
    pf_mkv::ebml::write_uint(&mut out, element_id, v);
    out
}

fn mkv_utf8(element_id: &[u8], v: &str) -> Vec<u8> {
    let mut out = Vec::new();
    pf_mkv::ebml::write_string(&mut out, element_id, v);
    out
}

fn mkv_binary(element_id: &[u8], v: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    pf_mkv::ebml::write_binary(&mut out, element_id, v);
    out
}

/// `\Segment\Info` (RFC 9559 §5.1.2): the TimestampScale in ns per tick, and the Duration as
/// a float in those ticks.
pub fn mkv_info(scale: u64, duration_ticks: f64) -> Vec<u8> {
    use pf_mkv::ebml::id;
    let mut dur = Vec::new();
    pf_mkv::ebml::write_f64(&mut dur, id::DURATION, duration_ticks);
    mkv_master(id::INFO, &[mkv_uint(id::TIMESTAMP_SCALE, scale), dur])
}

/// `\Segment\Tracks` (RFC 9559 §5.1.4) with a single audio TrackEntry (TrackType 2).
pub fn mkv_tracks(rate: f64, channels: u64) -> Vec<u8> {
    use pf_mkv::ebml::id;
    let mut freq = Vec::new();
    pf_mkv::ebml::write_f64(&mut freq, id::SAMPLING_FREQUENCY, rate);
    let audio = mkv_master(id::AUDIO, &[freq, mkv_uint(id::CHANNELS, channels)]);
    let entry = mkv_master(
        id::TRACK_ENTRY,
        &[mkv_uint(id::TRACK_NUMBER, 1), mkv_uint(id::TRACK_TYPE, 2), mkv_utf8(id::CODEC_ID, "A_OPUS"), audio],
    );
    mkv_master(id::TRACKS, &[entry])
}

/// `\Segment\Tags` (RFC 9559 §5.1.8): one untargeted `Tag` whose `SimpleTag`s are `entries`.
/// Untargeted is the shape ffmpeg writes and the one that means "this recording".
pub fn mkv_tags(entries: &[(&str, &str)]) -> Vec<u8> {
    use pf_mkv::ebml::id;
    let simples: Vec<Vec<u8>> = entries
        .iter()
        .map(|(name, value)| {
            mkv_master(id::SIMPLE_TAG, &[mkv_utf8(id::TAG_NAME, name), mkv_utf8(id::TAG_STRING, value)])
        })
        .collect();
    let mut kids = vec![mkv_master(id::TARGETS, &[])];
    kids.extend(simples);
    mkv_master(id::TAGS, &[mkv_master(id::TAG, &kids)])
}

/// `\Segment\Attachments` (RFC 9559 §5.1.7) holding one `cover.*` image — the cover-art
/// convention ([MatroskaTags], "Cover Art").
pub fn mkv_attachments(mime: &str, data: &[u8]) -> Vec<u8> {
    use pf_mkv::ebml::id;
    let ext = mime.rsplit('/').next().unwrap_or("jpeg");
    let file = mkv_master(
        id::ATTACHED_FILE,
        &[
            mkv_utf8(id::FILE_NAME, &format!("cover.{ext}")),
            mkv_utf8(id::FILE_MEDIA_TYPE, mime),
            mkv_binary(id::FILE_DATA, data),
            mkv_uint(id::FILE_UID, 0xC0FFEE),
        ],
    );
    mkv_master(id::ATTACHMENTS, &[file])
}

/// A `\Segment\SeekHead` (RFC 9559 §5.1.1) of `(target element ID, Segment Position)` entries.
/// Positions are relative to the first byte of the Segment's *data* (§4).
pub fn mkv_seek_head(entries: &[(&[u8], u64)]) -> Vec<u8> {
    use pf_mkv::ebml::id;
    let seeks: Vec<Vec<u8>> = entries
        .iter()
        .map(|(target, pos)| {
            mkv_master(id::SEEK, &[mkv_binary(id::SEEK_ID, target), mkv_uint(id::SEEK_POSITION, *pos)])
        })
        .collect();
    mkv_master(id::SEEK_HEAD, &seeks)
}

/// A `Cluster` of `n` filler bytes in one SimpleBlock — a stand-in for the frames a real file
/// puts between its front index and any trailing metadata.
pub fn mkv_cluster(n: usize) -> Vec<u8> {
    use pf_mkv::ebml::id;
    mkv_master(id::CLUSTER, &[mkv_uint(id::TIMESTAMP, 0), mkv_binary(id::SIMPLE_BLOCK, &vec![0u8; n])])
}

/// The EBML Header (RFC 8794 §11.2.4) declaring the given DocType.
pub fn mkv_ebml_header(doc_type: &str) -> Vec<u8> {
    use pf_mkv::ebml::id;
    mkv_master(
        id::EBML,
        &[
            mkv_uint(id::EBML_VERSION, 1),
            mkv_uint(id::EBML_READ_VERSION, 1),
            mkv_utf8(id::DOC_TYPE, doc_type),
            mkv_uint(id::DOC_TYPE_VERSION, 4),
            mkv_uint(id::DOC_TYPE_READ_VERSION, 2),
        ],
    )
}

/// Wrap level-1 masters in an EBML Header + a definite-size Segment. Returns the file bytes
/// and the absolute offset of the Segment's data (the base for every Segment Position).
pub fn mkv_segment(doc_type: &str, level1: &[Vec<u8>]) -> (Vec<u8>, u64) {
    let header = mkv_ebml_header(doc_type);
    let body: Vec<u8> = level1.concat();
    let mut seg = Vec::new();
    pf_mkv::ebml::write_id(&mut seg, pf_mkv::ebml::id::SEGMENT);
    pf_mkv::ebml::write_size(&mut seg, body.len() as u64);
    let data_start = (header.len() + seg.len()) as u64;
    let mut out = header;
    out.extend_from_slice(&seg);
    out.extend_from_slice(&body);
    (out, data_start)
}

/// A complete WebM file with everything **ahead of the frames** — the layout every one of the
/// reference library's 24 `.webm` files has, and the one a scanner reads in a single op.
///
/// The SeekHead indexes Info, Tracks and (when there are any) Tags, exactly as a muxer's
/// finalize writes it.
pub fn mkv(
    rate: f64,
    channels: u64,
    duration_ticks: f64,
    tag_entries: &[(&str, &str)],
    picture: Option<(&str, &[u8])>,
) -> Vec<u8> {
    use pf_mkv::ebml::id;
    let info = mkv_info(1_000_000, duration_ticks);
    let tracks = mkv_tracks(rate, channels);
    let tags = (!tag_entries.is_empty()).then(|| mkv_tags(tag_entries));
    let attachments = picture.map(|(mime, data)| mkv_attachments(mime, data));

    // The SeekHead's own length depends on the positions it carries, which depend on its
    // length — settle both by iteration, the fixed point a real finalize reaches by reserving
    // the space up front. The VINT width only grows, so a few rounds converge.
    let mut positions = [0u64; 4];
    let mut head = Vec::new();
    for _ in 0..8 {
        let mut entries: Vec<(&[u8], u64)> =
            vec![(id::INFO, positions[0]), (id::TRACKS, positions[1])];
        if tags.is_some() {
            entries.push((id::TAGS, positions[2]));
        }
        if attachments.is_some() {
            entries.push((id::ATTACHMENTS, positions[3]));
        }
        head = mkv_seek_head(&entries);
        let mut at = head.len() as u64;
        positions[0] = at;
        at += info.len() as u64;
        positions[1] = at;
        at += tracks.len() as u64;
        positions[2] = at;
        at += tags.as_ref().map_or(0, |t| t.len() as u64);
        positions[3] = at;
    }

    let mut level1 = vec![head, info, tracks];
    level1.extend(tags);
    level1.extend(attachments);
    level1.push(mkv_cluster(256));
    mkv_segment("webm", &level1).0
}

/// A Matroska file whose `Tags` and `Attachments` are written **behind** `filler` bytes of
/// frames — the layout a single-pass muxer produces, and the one that makes the SeekHead worth
/// having: the metadata is only reachable through the positions it names.
pub fn mkv_trailing(
    rate: f64,
    channels: u64,
    duration_ticks: f64,
    tag_entries: &[(&str, &str)],
    picture: Option<(&str, &[u8])>,
    filler: usize,
) -> Vec<u8> {
    use pf_mkv::ebml::id;
    let info = mkv_info(1_000_000, duration_ticks);
    let tracks = mkv_tracks(rate, channels);
    let tags = mkv_tags(tag_entries);
    let attachments = picture.map(|(mime, data)| mkv_attachments(mime, data));
    let cluster = mkv_cluster(filler);

    let (mut tags_pos, mut att_pos) = (0u64, 0u64);
    let mut head = Vec::new();
    for _ in 0..8 {
        let mut entries: Vec<(&[u8], u64)> = vec![(id::INFO, 0), (id::TRACKS, 0), (id::TAGS, tags_pos)];
        if attachments.is_some() {
            entries.push((id::ATTACHMENTS, att_pos));
        }
        head = mkv_seek_head(&entries);
        // Info and Tracks sit in the window anyway, so only the trailing positions must be
        // right; the parser prefers an element it can see over a SeekHead's claim about it.
        tags_pos = (head.len() + info.len() + tracks.len() + cluster.len()) as u64;
        att_pos = tags_pos + tags.len() as u64;
    }

    let mut level1 = vec![head, info, tracks, cluster, tags];
    level1.extend(attachments);
    mkv_segment("matroska", &level1).0
}

/// A unique scratch directory under the system temp dir, created if absent. No `tempfile`
/// crate: the workspace stays dependency-light, and a scan fixture is a directory of files,
/// not a resource that needs a destructor.
pub fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf_tags_{}_{}", tag, std::process::id()));
    std::fs::create_dir_all(&p).expect("create scratch dir");
    p
}
