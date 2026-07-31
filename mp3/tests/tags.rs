//! `pf_mp3::id3` — ID3v2 / ID3v1 / APEv2 parsing, driven by hand-built byte fixtures.
//!
//! Every tag here is assembled byte-for-byte from the specifications (id3v2.4.0-structure
//! / id3v2.4.0-frames, id3v2.3.0, id3v2-00, the ID3v1 note, the APEv2 specification), not
//! read from a file: a committed `.mp3` would only prove that *one* tagger's output
//! parses, while a builder can produce the shapes real files carry but fixtures rarely do
//! — UTF-16 in both endiannesses, per-frame unsynchronisation across an `$FF $00` run in
//! cover art, v2.4 multi-value text, an APE tag hiding behind an ID3v1 block.
//!
//! The last test in each group is a truncation sweep: every prefix of a fixture, from zero
//! bytes to the whole thing, must parse without panicking (spec: Supervision — a corrupt
//! side-car may never take a stream down).

// Tests own their fixtures and their collected results outright; the allocation discipline
// clippy.toml enforces is about `process()` hot paths, not fixture construction.
#![allow(clippy::disallowed_methods)]

use pf_mp3::id3::{ape_tail_len, parse_ape, parse_v1, parse_v2, v2_total_len};
use profluens_core::event::TagSink;
use profluens_core::memory::Arena;

// ---------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------

/// A [`TagSink`] that keeps everything it is handed, so a test can assert on order and
/// repetition as well as content.
#[derive(Default)]
struct Collect {
    text: Vec<(String, String)>,
    pictures: Vec<(String, Vec<u8>)>,
}

impl TagSink for Collect {
    fn text(&mut self, key: &str, value: &str) {
        self.text.push((key.to_string(), value.to_string()));
    }
    fn picture(&mut self, mime: &str, data: &[u8]) {
        self.pictures.push((mime.to_string(), data.to_vec()));
    }
}

impl Collect {
    /// First value for `key` — the same precedence `TagList::get` applies.
    fn get(&self, key: &str) -> Option<&str> {
        self.text.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
    /// Every value for `key`, in emission order.
    fn all(&self, key: &str) -> Vec<&str> {
        self.text.iter().filter(|(k, _)| k == key).map(|(_, v)| v.as_str()).collect()
    }
}

fn arena() -> Arena {
    Arena::new(Arena::DEFAULT_CHUNK)
}

/// A 4-byte synchsafe integer (id3v2.4.0-structure §6.2).
fn synchsafe(n: u32) -> [u8; 4] {
    [((n >> 21) & 0x7F) as u8, ((n >> 14) & 0x7F) as u8, ((n >> 7) & 0x7F) as u8, (n & 0x7F) as u8]
}

/// An ID3v2 tag: `"ID3"`, major/revision version, flags, synchsafe body size, body (§3.1).
fn v2_tag(major: u8, flags: u8, body: &[u8]) -> Vec<u8> {
    let mut out = b"ID3".to_vec();
    out.extend_from_slice(&[major, 0, flags]);
    out.extend_from_slice(&synchsafe(body.len() as u32));
    out.extend_from_slice(body);
    out
}

/// An ID3v2.2 frame: 3-character ID, 24-bit big-endian size, no flags (`id3v2-00` §3.2).
fn f22(id: &[u8; 3], data: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&(data.len() as u32).to_be_bytes()[1..]);
    out.extend_from_slice(data);
    out
}

/// An ID3v2.3 frame: 4-character ID, 32-bit big-endian size, 2 flag bytes (§3.3).
fn f23(id: &[u8; 4], flags: [u8; 2], data: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(&flags);
    out.extend_from_slice(data);
    out
}

/// An ID3v2.4 frame: 4-character ID, **synchsafe** size, 2 flag bytes (§4). The size is the
/// data as stored — after unsynchronisation, not before.
fn f24(id: &[u8; 4], flags: [u8; 2], data: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&synchsafe(data.len() as u32));
    out.extend_from_slice(&flags);
    out.extend_from_slice(data);
    out
}

/// Apply the unsynchronisation scheme (§6.1): insert `$00` after every `$FF` that is
/// followed by `$00` or by a byte with its top three bits set — i.e. after every `$FF` that
/// could otherwise begin a false MPEG frame sync — and after a trailing `$FF`.
fn unsync(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for (i, &b) in src.iter().enumerate() {
        out.push(b);
        if b == 0xFF {
            match src.get(i + 1) {
                None => out.push(0),
                Some(&n) if n == 0x00 || n & 0xE0 == 0xE0 => out.push(0),
                Some(_) => {}
            }
        }
    }
    out
}

/// UTF-16 bytes with a leading BOM, in the requested endianness (RFC 2781 §3.2).
fn utf16_bom(s: &str, big_endian: bool) -> Vec<u8> {
    let mut out = if big_endian { vec![0xFE, 0xFF] } else { vec![0xFF, 0xFE] };
    for u in s.encode_utf16() {
        out.extend_from_slice(&if big_endian { u.to_be_bytes() } else { u.to_le_bytes() });
    }
    out
}

// ---------------------------------------------------------------------------------------
// ID3v2 — header sizing
// ---------------------------------------------------------------------------------------

#[test]
fn v2_total_len_with_and_without_footer() {
    // 10-byte header + 40-byte body.
    assert_eq!(v2_total_len(&v2_tag(4, 0x00, &[0u8; 40])), Some(50));
    // Flags bit 4 adds the 10-byte footer (§3.4), which the size field does not cover.
    assert_eq!(v2_total_len(&v2_tag(4, 0x10, &[0u8; 40])), Some(60));

    // Not a tag: audio bytes, and a strict prefix of the header.
    assert_eq!(v2_total_len(b"\xFF\xFB\x90\x00audio"), None);
    assert_eq!(v2_total_len(b"ID3\x04\x00\x00\x00"), None);
    assert_eq!(v2_total_len(b""), None);

    // §6.2: a size byte with its high bit set is not synchsafe, so this is not a tag
    // header — a chance "ID3" run inside audio must not swallow the stream.
    let mut bad = v2_tag(4, 0x00, &[0u8; 40]);
    bad[7] = 0x80;
    assert_eq!(v2_total_len(&bad), None);

    // §3.1: neither version byte is ever $FF.
    let mut bad = v2_tag(4, 0x00, &[0u8; 40]);
    bad[3] = 0xFF;
    assert_eq!(v2_total_len(&bad), None);
}

// ---------------------------------------------------------------------------------------
// ID3v2 — text
// ---------------------------------------------------------------------------------------

#[test]
fn v23_utf16_title_parses_in_both_endiannesses() {
    // Non-ASCII plus a scalar above the BMP, so the run exercises the 2-byte Latin-range
    // path, the 3-byte BMP path and surrogate pairing all at once.
    const TITLE: &str = "Bjørk 𝄞";
    let a = arena();

    for big_endian in [false, true] {
        let mut data = vec![0x01]; // $01 — UTF-16 with BOM
        data.extend_from_slice(&utf16_bom(TITLE, big_endian));
        let tag = v2_tag(3, 0x00, &f23(b"TIT2", [0, 0], &data));

        let mut sink = Collect::default();
        parse_v2(&tag, &a, &mut sink);
        assert_eq!(sink.get("TITLE"), Some(TITLE), "big_endian={big_endian}");
    }
}

#[test]
fn v24_utf16be_without_bom_and_utf8_text() {
    let a = arena();
    // $02 — UTF-16BE, no BOM (v2.4 only).
    let mut be = vec![0x02];
    for u in "Ångström".encode_utf16() {
        be.extend_from_slice(&u.to_be_bytes());
    }
    // $03 — UTF-8, borrowed straight out of the tag.
    let mut utf8 = vec![0x03];
    utf8.extend_from_slice("Étude".as_bytes());

    let mut body = f24(b"TALB", [0, 0], &be);
    body.extend_from_slice(&f24(b"TCOM", [0, 0], &utf8));
    let mut sink = Collect::default();
    parse_v2(&v2_tag(4, 0x00, &body), &a, &mut sink);

    assert_eq!(sink.get("ALBUM"), Some("Ångström"));
    assert_eq!(sink.get("COMPOSER"), Some("Étude"));
}

#[test]
fn v22_tt2_title_and_three_byte_sizes() {
    let a = arena();
    // ISO-8859-1: $E9 is é, which has to be transcoded into the arena.
    let mut body = f22(b"TT2", b"\x00Caf\xE9");
    body.extend_from_slice(&f22(b"TP1", b"\x00Nina Simone"));
    body.extend_from_slice(&f22(b"TRK", b"\x003/12"));

    let mut sink = Collect::default();
    parse_v2(&v2_tag(2, 0x00, &body), &a, &mut sink);

    assert_eq!(sink.get("TITLE"), Some("Café"));
    assert_eq!(sink.get("ARTIST"), Some("Nina Simone"));
    // Values pass through raw — "3/12" is a legal TRCK and stays one string.
    assert_eq!(sink.get("TRACKNUMBER"), Some("3/12"));
}

#[test]
fn v24_multi_value_text_splits_but_v23_does_not() {
    let a = arena();
    // §4.2: a v2.4 text frame may carry several values separated by the terminator.
    let data = b"\x03Rone\x00Bachar Mar-Khalif\x00Vanessa Wagner";

    let mut sink = Collect::default();
    parse_v2(&v2_tag(4, 0x00, &f24(b"TPE1", [0, 0], data)), &a, &mut sink);
    assert_eq!(sink.all("ARTIST"), ["Rone", "Bachar Mar-Khalif", "Vanessa Wagner"]);

    // The same bytes in a v2.3 tag are one value; everything past the first terminator is
    // padding the spec does not define, so it must not become extra artists.
    let mut sink = Collect::default();
    parse_v2(&v2_tag(3, 0x00, &f23(b"TPE1", [0, 0], data)), &a, &mut sink);
    assert_eq!(sink.all("ARTIST"), ["Rone"]);
}

#[test]
fn txxx_description_becomes_the_key_carrying_replaygain() {
    let a = arena();
    let mut body = f24(b"TXXX", [0, 0], b"\x00replaygain_track_gain\x00-6.40 dB");
    body.extend_from_slice(&f24(b"TXXX", [0, 0], b"\x00REPLAYGAIN_TRACK_PEAK\x000.977661"));
    body.extend_from_slice(&f24(b"TXXX", [0, 0], b"\x00replaygain_album_gain\x00-7.19 dB"));

    let mut sink = Collect::default();
    parse_v2(&v2_tag(4, 0x00, &body), &a, &mut sink);

    assert_eq!(sink.get("REPLAYGAIN_TRACK_GAIN"), Some("-6.40 dB"));
    assert_eq!(sink.get("REPLAYGAIN_TRACK_PEAK"), Some("0.977661"));
    assert_eq!(sink.get("REPLAYGAIN_ALBUM_GAIN"), Some("-7.19 dB"));
}

#[test]
fn comment_frame_skips_language_and_description() {
    let a = arena();
    // §4.10: encoding, 3-byte ISO-639-2 language, terminated short description, text.
    let v24 = f24(b"COMM", [0, 0], b"\x00engmood\x00Nocturnal, patient");
    let mut sink = Collect::default();
    parse_v2(&v2_tag(4, 0x00, &v24), &a, &mut sink);
    assert_eq!(sink.get("COMMENT"), Some("Nocturnal, patient"));

    // The v2.2 spelling, with an empty description.
    let v22 = f22(b"COM", b"\x00eng\x00Ripped from vinyl");
    let mut sink = Collect::default();
    parse_v2(&v2_tag(2, 0x00, &v22), &a, &mut sink);
    assert_eq!(sink.get("COMMENT"), Some("Ripped from vinyl"));
}

#[test]
fn padding_stops_the_frame_walk() {
    let a = arena();
    let mut body = f24(b"TIT2", [0, 0], b"\x03Reached");
    // §3.1: the tag may end in zero padding, and no frame ID starts with $00.
    body.extend_from_slice(&[0u8; 24]);
    // A perfectly well-formed frame *after* the padding must never be read: the padding is
    // the end of the frames, and walking into it would read arbitrary zero-derived sizes.
    body.extend_from_slice(&f24(b"TPE1", [0, 0], b"\x03Unreachable"));

    let mut sink = Collect::default();
    parse_v2(&v2_tag(4, 0x00, &body), &a, &mut sink);
    assert_eq!(sink.get("TITLE"), Some("Reached"));
    assert_eq!(sink.get("ARTIST"), None);
}

#[test]
fn unreadable_and_unknown_frames_are_skipped_not_fatal() {
    let a = arena();
    // v2.4 frame format flags (§4.1): compression $08, encryption $04, grouping $40.
    let mut body = f24(b"TALB", [0, 0x08], b"\x00compressed");
    body.extend_from_slice(&f24(b"TCOM", [0, 0x04], b"\x00encrypted"));
    body.extend_from_slice(&f24(b"TPOS", [0, 0x40], b"\x00grouped"));
    body.extend_from_slice(&f24(b"WOAR", [0, 0], b"https://example.invalid")); // unknown ID
    // ...and the frame after all of them still parses, which is the point.
    body.extend_from_slice(&f24(b"TIT2", [0, 0], b"\x00Survivor"));

    let mut sink = Collect::default();
    parse_v2(&v2_tag(4, 0x00, &body), &a, &mut sink);
    assert_eq!(sink.get("TITLE"), Some("Survivor"));
    assert_eq!(sink.text.len(), 1);
}

#[test]
fn extended_headers_are_stepped_over() {
    let a = arena();
    // v2.3 §3.2: a 32-bit size that *excludes* its own four bytes (6 = flags + padding).
    let mut ext23 = 6u32.to_be_bytes().to_vec();
    ext23.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
    ext23.extend_from_slice(&f23(b"TIT2", [0, 0], b"\x00After ext v2.3"));
    let mut sink = Collect::default();
    parse_v2(&v2_tag(3, 0x40, &ext23), &a, &mut sink);
    assert_eq!(sink.get("TITLE"), Some("After ext v2.3"));

    // v2.4 §3.2: a synchsafe size that *includes* its own four bytes.
    let mut ext24 = synchsafe(10).to_vec();
    ext24.extend_from_slice(&[0u8; 6]);
    ext24.extend_from_slice(&f24(b"TIT2", [0, 0], b"\x00After ext v2.4"));
    let mut sink = Collect::default();
    parse_v2(&v2_tag(4, 0x40, &ext24), &a, &mut sink);
    assert_eq!(sink.get("TITLE"), Some("After ext v2.4"));
}

// ---------------------------------------------------------------------------------------
// ID3v2 — unsynchronisation
// ---------------------------------------------------------------------------------------

/// An `APIC` frame body (§4.14) whose image data is deliberately full of the byte patterns
/// unsynchronisation exists to hide: `$FF $00`, `$FF $Ex`, `$FF $FF`, and a trailing `$FF`.
fn apic_body(image: &[u8]) -> Vec<u8> {
    let mut data = vec![0x00]; // $00 — ISO-8859-1 description
    data.extend_from_slice(b"image/jpeg\0");
    data.push(3); // picture type: front cover
    data.extend_from_slice(b"cover\0");
    data.extend_from_slice(image);
    data
}

const ART: &[u8] = &[
    0xFF, 0xD8, 0xFF, 0x00, 0xFF, 0xE0, 0x00, 0x10, 0xFF, 0xFF, 0x00, 0x00, 0x42, 0xFF,
];

#[test]
fn v24_per_frame_unsync_apic_round_trips_byte_exactly() {
    let a = arena();
    let data = apic_body(ART);
    let stored = unsync(&data);
    assert_ne!(stored, data, "the fixture must actually be unsynchronised");
    // §4.1: the v2.4 frame size counts the data as stored, and flag bit $02 marks it.
    let tag = v2_tag(4, 0x00, &f24(b"APIC", [0, 0x02], &stored));

    let mut sink = Collect::default();
    parse_v2(&tag, &a, &mut sink);
    assert_eq!(sink.pictures.len(), 1);
    assert_eq!(sink.pictures[0].0, "image/jpeg");
    assert_eq!(sink.pictures[0].1, ART);
}

#[test]
fn v24_tag_level_unsync_flag_applies_to_every_frame() {
    let a = arena();
    // §3.1: in v2.4 the tag-level bit asserts that all frames are unsynchronised, so the
    // per-frame flag may legitimately be absent even though the data is encoded.
    let tag = v2_tag(4, 0x80, &f24(b"APIC", [0, 0x00], &unsync(&apic_body(ART))));
    let mut sink = Collect::default();
    parse_v2(&tag, &a, &mut sink);
    assert_eq!(sink.pictures.len(), 1);
    assert_eq!(sink.pictures[0].1, ART);
}

#[test]
fn v23_whole_tag_unsync_covers_the_frame_headers_too() {
    let a = arena();
    // v2.3 unsynchronises the assembled body — frame headers included — so the frame sizes
    // stored are the *pre*-unsynchronisation ones and the tag size is the post one.
    let mut body = f23(b"APIC", [0, 0], &apic_body(ART));
    body.extend_from_slice(&f23(b"TIT2", [0, 0], b"\x00Cover test"));
    let tag = v2_tag(3, 0x80, &unsync(&body));

    let mut sink = Collect::default();
    parse_v2(&tag, &a, &mut sink);
    assert_eq!(sink.pictures.len(), 1);
    assert_eq!(sink.pictures[0].1, ART);
    assert_eq!(sink.get("TITLE"), Some("Cover test"));
}

#[test]
fn v22_pic_maps_the_three_character_format_code() {
    let a = arena();
    // `id3v2-00` §4.15: `PIC` carries a 3-character image format, not a MIME type.
    let mut data = vec![0x00];
    data.extend_from_slice(b"PNG");
    data.push(3);
    data.extend_from_slice(b"\x00"); // empty description
    data.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    let mut sink = Collect::default();
    parse_v2(&v2_tag(2, 0x00, &f22(b"PIC", &data)), &a, &mut sink);
    assert_eq!(sink.pictures.len(), 1);
    assert_eq!(sink.pictures[0].0, "image/png");
    assert_eq!(sink.pictures[0].1, b"\x89PNG\r\n\x1a\n");
}

// ---------------------------------------------------------------------------------------
// ID3v1
// ---------------------------------------------------------------------------------------

/// A 128-byte ID3v1 record. `comment` is written into the 30-byte field verbatim, so a
/// caller can build both the v1.0 and the v1.1 (track-number) shapes.
fn v1_tag(title: &[u8], artist: &[u8], album: &[u8], year: &[u8], comment: &[u8], genre: u8) -> Vec<u8> {
    let mut out = b"TAG".to_vec();
    let mut field = |src: &[u8], n: usize| {
        let mut f = vec![0u8; n];
        f[..src.len().min(n)].copy_from_slice(&src[..src.len().min(n)]);
        out.extend_from_slice(&f);
    };
    field(title, 30);
    field(artist, 30);
    field(album, 30);
    field(year, 4);
    field(comment, 30);
    out.push(genre);
    assert_eq!(out.len(), 128);
    out
}

#[test]
fn id3v1_1_reads_the_track_number_out_of_the_comment() {
    let a = arena();
    // v1.1: comment[28] == $00 and comment[29] != $00 means the last byte is the track.
    let mut comment = vec![0u8; 30];
    comment[..7].copy_from_slice(b"Vinyl B");
    comment[29] = 7;
    let tag = v1_tag(b"Sad Song", b"Caf\xE9 del Mar", b"Aria", b"1994", &comment, 17);

    let mut sink = Collect::default();
    parse_v1(&tag, &a, &mut sink);

    assert_eq!(sink.get("TITLE"), Some("Sad Song"));
    assert_eq!(sink.get("ARTIST"), Some("Café del Mar")); // ISO-8859-1 → UTF-8
    assert_eq!(sink.get("ALBUM"), Some("Aria"));
    assert_eq!(sink.get("DATE"), Some("1994"));
    assert_eq!(sink.get("COMMENT"), Some("Vinyl B"));
    assert_eq!(sink.get("TRACKNUMBER"), Some("7"));
    assert_eq!(sink.get("GENRE"), Some("Rock")); // index 17
}

#[test]
fn id3v1_0_keeps_all_thirty_comment_bytes_and_skips_empty_fields() {
    let a = arena();
    // A full 30-byte comment: comment[28] is non-zero, so this is v1.0 and there is no
    // track number to steal.
    let comment = b"012345678901234567890123456789";
    // Space-padded fields (some taggers) and an empty album must not emit blanks.
    let tag = v1_tag(b"Title   ", b"   ", b"", b"2019", comment, 0xFF);

    let mut sink = Collect::default();
    parse_v1(&tag, &a, &mut sink);

    assert_eq!(sink.get("TITLE"), Some("Title"));
    assert_eq!(sink.get("ARTIST"), None);
    assert_eq!(sink.get("ALBUM"), None);
    assert_eq!(sink.get("COMMENT"), Some("012345678901234567890123456789"));
    assert_eq!(sink.get("TRACKNUMBER"), None);
    assert_eq!(sink.get("GENRE"), None); // $FF is past the table — unset
}

#[test]
fn id3v1_genre_table_is_indexed_correctly_at_its_boundaries() {
    let a = arena();
    // An off-by-one anywhere in a 148-entry literal shows up at the seams: the end of the
    // original 0..=79 list, the start of the Winamp extension, and its last index.
    for (index, name) in [(0u8, "Blues"), (79, "Hard Rock"), (80, "Folk"), (147, "Synthpop")] {
        let mut sink = Collect::default();
        parse_v1(&v1_tag(b"t", b"", b"", b"", b"", index), &a, &mut sink);
        assert_eq!(sink.get("GENRE"), Some(name), "genre index {index}");
    }
    // 148 is the first index past the extension.
    let mut sink = Collect::default();
    parse_v1(&v1_tag(b"t", b"", b"", b"", b"", 148), &a, &mut sink);
    assert_eq!(sink.get("GENRE"), None);
}

#[test]
fn v2_is_emitted_before_v1_so_it_wins_on_conflict() {
    let a = arena();
    let v2 = v2_tag(4, 0x00, &f24(b"TIT2", [0, 0], b"\x03A Title Longer Than Thirty Characters"));
    let v1 = v1_tag(b"A Title Longer Than Thirty Cha", b"", b"", b"", b"", 0xFF);

    // The documented call order: v2 first, then v1 — `TagList::get` takes the first value.
    let mut sink = Collect::default();
    parse_v2(&v2, &a, &mut sink);
    parse_v1(&v1, &a, &mut sink);
    assert_eq!(sink.get("TITLE"), Some("A Title Longer Than Thirty Characters"));
    // Both are still recorded; only the lookup order decides.
    assert_eq!(sink.all("TITLE").len(), 2);
}

// ---------------------------------------------------------------------------------------
// APEv2
// ---------------------------------------------------------------------------------------

/// One APEv2 item: `u32` LE value length, `u32` LE item flags, NUL-terminated ASCII key,
/// value bytes.
fn ape_item(key: &str, value: &[u8], flags: u32) -> Vec<u8> {
    let mut out = (value.len() as u32).to_le_bytes().to_vec();
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(key.as_bytes());
    out.push(0);
    out.extend_from_slice(value);
    out
}

/// An APEv2 tag: optional 32-byte header, the items, then the mandatory 32-byte footer.
/// The tag size in both blocks covers the items plus the footer, never the header.
fn ape_tag(items: &[Vec<u8>], with_header: bool) -> Vec<u8> {
    let body: Vec<u8> = items.concat();
    let size = (body.len() + 32) as u32;
    let has_header = if with_header { 1u32 << 31 } else { 0 };
    let block = |flags: u32| {
        let mut b = b"APETAGEX".to_vec();
        b.extend_from_slice(&2000u32.to_le_bytes()); // APEv2
        b.extend_from_slice(&size.to_le_bytes());
        b.extend_from_slice(&(items.len() as u32).to_le_bytes());
        b.extend_from_slice(&flags.to_le_bytes());
        b.extend_from_slice(&[0u8; 8]); // reserved
        b
    };
    let mut out = Vec::new();
    if with_header {
        out.extend_from_slice(&block(has_header | (1 << 29))); // bit 29: this is the header
    }
    out.extend_from_slice(&body);
    out.extend_from_slice(&block(has_header));
    out
}

fn replaygain_items() -> Vec<Vec<u8>> {
    vec![
        ape_item("replaygain_track_gain", b"-6.40 dB", 0),
        ape_item("replaygain_track_peak", b"0.977661", 0),
        ape_item("replaygain_album_gain", b"-7.19 dB", 0),
        ape_item("Title", b"Corner", 0),
        ape_item("Year", b"2019", 0),                       // → DATE
        ape_item("Track", b"4/9", 0),                       // → TRACKNUMBER
        ape_item("Artist", b"Sofia\x00Emil", 0),            // NUL-separated multi-value
        ape_item("Cover Art (Front)", b"\xFF\xD8\xFF\xE0", 2), // binary — must be skipped
    ]
}

#[test]
fn ape_tail_len_measures_the_tag_from_its_footer() {
    let tag = ape_tag(&replaygain_items(), false);
    assert_eq!(ape_tail_len(&tag[tag.len() - 32..]), Some(tag.len()));

    // With a header the tag is 32 bytes longer, but the size field is unchanged — the
    // header-present flag is what accounts for it.
    let tag = ape_tag(&replaygain_items(), true);
    assert_eq!(ape_tail_len(&tag[tag.len() - 32..]), Some(tag.len()));

    // Not a footer: audio bytes, a short slice, and an APE *header* at the end.
    assert_eq!(ape_tail_len(&[0u8; 32]), None);
    assert_eq!(ape_tail_len(b"APETAGEX"), None);
    let header_only = &ape_tag(&replaygain_items(), true)[..32];
    assert_eq!(ape_tail_len(header_only), None);
}

#[test]
fn ape_text_items_become_canonical_keys_and_binary_items_are_skipped() {
    for with_header in [false, true] {
        let tag = ape_tag(&replaygain_items(), with_header);
        let mut sink = Collect::default();
        parse_ape(&tag, &mut sink);

        assert_eq!(sink.get("REPLAYGAIN_TRACK_GAIN"), Some("-6.40 dB"));
        assert_eq!(sink.get("REPLAYGAIN_TRACK_PEAK"), Some("0.977661"));
        assert_eq!(sink.get("REPLAYGAIN_ALBUM_GAIN"), Some("-7.19 dB"));
        assert_eq!(sink.get("TITLE"), Some("Corner"));
        // APE's own key names mapped onto the canonical vocabulary.
        assert_eq!(sink.get("DATE"), Some("2019"));
        assert_eq!(sink.get("TRACKNUMBER"), Some("4/9"));
        // A NUL inside a text value separates two values.
        assert_eq!(sink.all("ARTIST"), ["Sofia", "Emil"]);
        // The binary item (flag bits 1-2 == 1) never reaches the sink.
        assert!(sink.text.iter().all(|(k, _)| k != "COVER ART (FRONT)"), "{with_header}");
        assert!(sink.pictures.is_empty());
    }
}

#[test]
fn ape_hides_behind_a_trailing_id3v1_block() {
    let a = arena();
    // The conventional on-disk order: … audio, APEv2, ID3v1, EOF.
    let ape = ape_tag(&replaygain_items(), false);
    let v1 = v1_tag(b"Corner", b"Sofia", b"", b"2019", b"", 26);
    let mut tail = ape.clone();
    tail.extend_from_slice(&v1);

    // Probing the literal last 32 bytes finds nothing — those are ID3v1's.
    assert_eq!(ape_tail_len(&tail[tail.len() - 32..]), None);
    // Strip the 128-byte ID3v1 block first and the footer is right there.
    let without_v1 = &tail[..tail.len() - 128];
    assert_eq!(ape_tail_len(&without_v1[without_v1.len() - 32..]), Some(ape.len()));

    // Both tags then read out of the same file tail, in the documented order.
    let mut sink = Collect::default();
    parse_ape(&without_v1[without_v1.len() - ape.len()..], &mut sink);
    parse_v1(&tail, &a, &mut sink);
    assert_eq!(sink.get("REPLAYGAIN_TRACK_GAIN"), Some("-6.40 dB"));
    assert_eq!(sink.get("TITLE"), Some("Corner")); // APE's, emitted first
    assert_eq!(sink.get("GENRE"), Some("Ambient")); // only ID3v1 carries it (index 26)
}

// ---------------------------------------------------------------------------------------
// Robustness
// ---------------------------------------------------------------------------------------

/// A tag exercising most of the parser at once, for the truncation sweeps.
fn kitchen_sink_v24() -> Vec<u8> {
    let mut body = f24(b"TIT2", [0, 0], b"\x03Kitchen Sink");
    body.extend_from_slice(&f24(b"TPE1", [0, 0], b"\x03One\x00Two"));
    body.extend_from_slice(&f24(b"TXXX", [0, 0], b"\x00replaygain_track_gain\x00-6.40 dB"));
    body.extend_from_slice(&f24(b"COMM", [0, 0], b"\x01eng\xFF\xFE\x00\x00\xFF\xFEh\x00i\x00"));
    body.extend_from_slice(&f24(b"APIC", [0, 0x02], &unsync(&apic_body(ART))));
    body.extend_from_slice(&[0u8; 8]); // padding
    v2_tag(4, 0x00, &body)
}

#[test]
fn every_prefix_of_a_tag_parses_without_panicking() {
    let mut a = arena();
    let v22 = {
        let mut body = f22(b"TT2", b"\x00Caf\xE9");
        body.extend_from_slice(&f22(b"PIC", b"\x00JPG\x03\x00\xFF\xD8\xFF\xE0"));
        v2_tag(2, 0x00, &body)
    };
    let v23 = v2_tag(3, 0x80, &unsync(&f23(b"APIC", [0, 0], &apic_body(ART))));
    let fixtures = [kitchen_sink_v24(), v23, v22];

    for tag in &fixtures {
        for n in 0..=tag.len() {
            let mut sink = Collect::default();
            parse_v2(&tag[..n], &a, &mut sink);
            v2_total_len(&tag[..n]);
            a.reset();
        }
        // Every single-byte corruption, at every offset, must also be survivable: this is
        // where a bogus frame size or a mid-UTF-16 cut would otherwise index out of range.
        for i in 0..tag.len() {
            for byte in [0x00u8, 0x7F, 0x80, 0xFF] {
                let mut corrupt = tag.clone();
                corrupt[i] = byte;
                let mut sink = Collect::default();
                parse_v2(&corrupt, &a, &mut sink);
                v2_total_len(&corrupt);
                a.reset();
            }
        }
    }
}

#[test]
fn every_prefix_of_a_v1_or_ape_tail_parses_without_panicking() {
    let mut a = arena();
    let v1 = v1_tag(b"T", b"A", b"B", b"2019", b"c", 17);
    let ape = ape_tag(&replaygain_items(), true);

    for n in 0..=v1.len() {
        let mut sink = Collect::default();
        parse_v1(&v1[..n], &a, &mut sink);
        a.reset();
    }
    for n in 0..=ape.len() {
        let mut sink = Collect::default();
        parse_ape(&ape[..n], &mut sink);
        ape_tail_len(&ape[..n]);
    }
    // A footer advertising a wildly oversized tag must not be believed into an overrun.
    let mut lying = ape.clone();
    let footer = lying.len() - 32;
    lying[footer + 12..footer + 16].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut sink = Collect::default();
    parse_ape(&lying, &mut sink);
    assert_eq!(ape_tail_len(&lying[lying.len() - 32..]), Some(u32::MAX as usize + 32));

    // Random-ish garbage in the shapes the magic bytes would admit.
    for tag in [b"ID3".as_slice(), b"TAG", b"APETAGEX"] {
        let mut noisy = tag.to_vec();
        noisy.extend((0..300u32).map(|i| (i.wrapping_mul(2654435761) >> 16) as u8));
        let mut sink = Collect::default();
        parse_v2(&noisy, &a, &mut sink);
        parse_v1(&noisy, &a, &mut sink);
        parse_ape(&noisy, &mut sink);
        a.reset();
    }
}
