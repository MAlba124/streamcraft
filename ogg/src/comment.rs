//! Vorbis comments: the metadata block Vorbis, Opus and FLAC all share (spec: Vorbis I
//! specification §5, "comment field and header specification"; RFC 7845 §5.2; RFC 9639 §8.6).
//!
//! Three codecs, one structure. Vorbis puts it in its second header packet behind
//! `0x03 "vorbis"`, Opus behind the magic `"OpusTags"`, and FLAC in a `VORBIS_COMMENT`
//! metadata block — but the bytes after that prefix are identical, so
//! [`parse_comment_body`] is the whole parser and [`parse_vorbis_comments`] /
//! [`parse_opus_tags`] are three-line wrappers that strip the codec's prefix.
//!
//! ## Allocation and borrowing
//! Nothing here touches the heap (spec: allocation discipline — no steady-state heap
//! traffic). Values are handed to the [`TagSink`] as slices **borrowed from `body`**; the
//! field name is uppercased into a stack buffer (or, for an absurdly long name, into the
//! caller's scratch [`Arena`]), and an embedded picture is base64-decoded into that same
//! arena. Everything the sink receives is valid for the call only — a sink that retains it
//! copies, which is exactly what `TagListSink` does.
//!
//! ## Hostile input
//! A tag block is attacker-controlled: every length is checked against the remaining bytes
//! and a malformed entry is skipped rather than aborting the walk (a truncated last entry
//! must not cost you the ten good ones before it). Nothing here can panic.

use profluens_core::event::TagSink;
use profluens_core::memory::Arena;

/// The comment field whose value is a base64-encoded FLAC `PICTURE` block. Defined by the
/// xiph "Ogg Embedded Metadata"/FLAC convention: cover art in a Vorbis comment is the FLAC
/// picture block (RFC 9639 §8.8) base64-encoded (RFC 4648 §4) into the field value, because
/// a comment value is textual. Compared case-insensitively, like every field name (Vorbis I
/// §5.2.2.1: "field names are not case sensitive").
const PICTURE_FIELD: &str = "METADATA_BLOCK_PICTURE";

/// Longest field name uppercased on the stack; longer ones fall back to the scratch arena.
/// Real field names are a handful of bytes (`TITLE`, `REPLAYGAIN_TRACK_GAIN`), so the arena
/// path exists only so hostile input cannot force an allocation *or* a truncation.
const KEY_STACK_MAX: usize = 64;

/// Parse the body of a Vorbis comment block — the bytes **after** any codec-specific magic
/// (Vorbis I §5.2.2.1):
///
/// ```text
///   u32le vendor_length | vendor_string (UTF-8, not terminated)
///   u32le user_comment_list_length
///   repeat: u32le length | "FIELD=value" (UTF-8, not terminated)
/// ```
///
/// Each entry is emitted as [`TagSink::text`] with the field name uppercased (the canonical
/// key vocabulary) and the value borrowed from `body`. A `METADATA_BLOCK_PICTURE` entry is
/// instead base64-decoded and emitted as [`TagSink::picture`].
///
/// The vendor string is skipped: it names the *encoder*, not the content, and has no field
/// name to key it by. A malformed entry (bad length, invalid UTF-8, no `=`, illegal field
/// name, undecodable picture) is skipped and the walk continues; a length that overruns the
/// body ends it. Repeated field names are emitted repeatedly, in file order — `ARTIST` twice
/// is how the format expresses two artists (Vorbis I §5.2.2.1).
pub fn parse_comment_body(body: &[u8], scratch: &Arena, sink: &mut impl TagSink) {
    let Some(vendor_len) = le_u32(body, 0) else { return };
    // 4 + vendor_len can overflow only on a 32-bit target; checked add keeps the walk honest
    // on both.
    let Some(count_at) = 4usize.checked_add(vendor_len) else { return };
    let Some(count) = le_u32(body, count_at) else { return };
    let mut pos = count_at + 4;

    let mut key_buf = [0u8; KEY_STACK_MAX];
    for _ in 0..count {
        let Some(len) = le_u32(body, pos) else { break };
        pos += 4;
        let Some(entry) = body.get(pos..).and_then(|rest| rest.get(..len)) else { break };
        pos += len;

        // "Field names and values are separated by an '=' character" (Vorbis I §5.2.2.1) —
        // the *first* one, so a value may itself contain '='.
        let Ok(entry) = std::str::from_utf8(entry) else { continue };
        let Some((raw_key, value)) = entry.split_once('=') else { continue };
        let Some(key) = upper_key(raw_key.as_bytes(), &mut key_buf, scratch) else { continue };

        if key == PICTURE_FIELD {
            if let Some(block) = base64_decode(value.as_bytes(), scratch) {
                if let Some((mime, data)) = parse_picture_block(block) {
                    sink.picture(mime, data);
                }
            }
            // A picture that will not decode is dropped, not emitted as text: a caller
            // asking for tags wants the image or nothing, never 40 KiB of base64 in a
            // `TITLE`-shaped slot.
            continue;
        }
        sink.text(key, value);
    }
}

/// Parse an Opus comment header packet: the magic `"OpusTags"` then a Vorbis comment body
/// (RFC 7845 §5.2 "Comment Header"). Returns false (having emitted nothing) if the magic is
/// absent — i.e. this is not the comment packet.
///
/// The optional trailing "user comment list" padding RFC 7845 §5.2 allows after the last
/// comment needs no handling: the walk is bounded by the comment count, so trailing bytes
/// are simply never read.
pub fn parse_opus_tags(packet: &[u8], scratch: &Arena, sink: &mut impl TagSink) -> bool {
    let Some(body) = packet.strip_prefix(b"OpusTags") else { return false };
    parse_comment_body(body, scratch, sink);
    true
}

/// Parse a Vorbis comment header packet: packet type `0x03` and the codec identifier
/// `"vorbis"` (Vorbis I §4.2.1 common header decode), then a Vorbis comment body
/// (§5.2.2.1). Returns false (having emitted nothing) if the packet is not a comment header.
///
/// The framing bit the spec appends after the comment list (§5.2.2.1: "the framing bit must
/// be nonzero") is deliberately ignored: it guards the *bit* packing of the Vorbis header
/// stream, and a decoder that refuses tags over one flipped bit at the end helps nobody.
pub fn parse_vorbis_comments(packet: &[u8], scratch: &Arena, sink: &mut impl TagSink) -> bool {
    let Some(body) = packet.strip_prefix(b"\x03vorbis") else { return false };
    parse_comment_body(body, scratch, sink);
    true
}

/// Read a little-endian u32 length at `pos` as a `usize` (Vorbis I §5.2.2.1: comment lengths
/// are little-endian, unlike the rest of FLAC's big-endian fields — RFC 9639 §8.6 calls that
/// out explicitly). `None` if fewer than four bytes remain.
fn le_u32(b: &[u8], pos: usize) -> Option<usize> {
    let s = b.get(pos..)?.get(..4)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
}

/// Uppercase a field name into `stack` (or the arena when it does not fit) and return it as
/// a `&str`. `None` if the name is empty or contains a byte outside the legal set.
///
/// Legal field-name bytes: printable ASCII excluding `=`. The two specs that define this
/// block disagree on the last character — Vorbis I §5.2.2.1 says 0x20–0x7D, RFC 9639 §8.6
/// says U+0020–U+007E — so this reader accepts the union (0x20..=0x7E, `=` excluded): a
/// field name a FLAC file may legally carry must not be dropped just because it ends up in
/// an Ogg stream, and the only byte in dispute is `~`.
///
/// Uppercasing is ASCII-only, which is exactly right: the name is ASCII by definition and
/// case-insensitive comparison is defined over A–Z/a–z (Vorbis I §5.2.2.1).
fn upper_key<'k>(raw: &[u8], stack: &'k mut [u8; KEY_STACK_MAX], scratch: &'k Arena) -> Option<&'k str> {
    if raw.is_empty() || !raw.iter().all(|&b| (0x20..=0x7E).contains(&b) && b != b'=') {
        return None;
    }
    let buf: &'k mut [u8] =
        if raw.len() <= KEY_STACK_MAX { &mut stack[..raw.len()] } else { scratch.alloc_bytes(raw.len()) };
    for (dst, &src) in buf.iter_mut().zip(raw) {
        *dst = src.to_ascii_uppercase();
    }
    // Every byte is printable ASCII (checked above), so this cannot fail; `from_utf8` keeps
    // the module free of `unsafe` (`#![deny(unsafe_code)]`) at the cost of one scan.
    std::str::from_utf8(buf).ok()
}

/// Decode base64 (RFC 4648 §4, the standard `A–Z a–z 0–9 + /` alphabet) into `scratch`.
/// `None` if the input is not decodable, so a corrupt picture field is skipped rather than
/// half-decoded.
///
/// Padding (§4: "the '=' character is used to signal the end of the data"): trailing `=`s are
/// stripped and then *required to be consistent* — a padded input's total length must be a
/// multiple of 4 and the pad count must match the remainder (2 data characters need `==`,
/// 3 need `=`). Unpadded input is accepted too (§3.2 allows padding to be omitted when the
/// data length is known, which it is here — the field length framed it), but a remainder of
/// 1 character is impossible in any encoding and is rejected. `=` anywhere but the tail is
/// rejected, as is any non-alphabet byte (§3.3: "implementations MUST reject the encoded
/// data if it contains characters outside the base alphabet") — including the whitespace
/// MIME-style base64 permits, which has no business inside a comment field.
///
/// Non-zero bits in the final partial group are tolerated (§3.5 leaves rejecting them
/// optional): the picture bytes they precede are still exactly recoverable.
fn base64_decode<'a>(input: &[u8], scratch: &'a Arena) -> Option<&'a [u8]> {
    let mut pad = 0;
    let mut n = input.len();
    while pad < 2 && n > 0 && input[n - 1] == b'=' {
        n -= 1;
        pad += 1;
    }
    let body = &input[..n];
    let rem = body.len() % 4;
    if rem == 1 || (pad > 0 && (rem + pad != 4)) {
        return None;
    }
    // Each full 4-character group is 3 bytes; a 2- or 3-character tail is 1 or 2 bytes.
    let out_len = body.len() / 4 * 3 + rem.saturating_sub(1);
    let out = scratch.alloc_bytes(out_len);

    // Four characters at a time, through the reverse-alphabet table: the 24 bits they carry
    // are assembled in one `u32` and written out as three bytes, so the per-character shift
    // bookkeeping (and its unpredictable "have I got 8 bits yet" branch) disappears. Rejection
    // is one test for the whole group — every out-of-alphabet byte maps to [`BAD`], whose two
    // high bits cannot appear in a legal 6-bit value, so OR-ing the four and masking `0xC0`
    // catches any of them at once. That also subsumes the "`=` may only end the input" rule
    // (§4) without the separate scan of the body it used to need: `DECODE[b'=']` is `BAD`.
    let full = body.len() / 4;
    let (main, tail) = out.split_at_mut(full * 3);
    // `as_chunks` rather than `chunks_exact`: the `[u8; 4]` / `[u8; 3]` element types carry
    // their length in the type, so every index below is checked at compile time instead of at
    // run time.
    let (groups, _) = body.as_chunks::<4>();
    let (outs, _) = main.as_chunks_mut::<3>();
    for (q, o) in groups.iter().zip(outs) {
        let a = DECODE[q[0] as usize] as u32;
        let b = DECODE[q[1] as usize] as u32;
        let c = DECODE[q[2] as usize] as u32;
        let d = DECODE[q[3] as usize] as u32;
        if (a | b | c | d) & 0xC0 != 0 {
            return None;
        }
        let v = (a << 18) | (b << 12) | (c << 6) | d;
        o[0] = (v >> 16) as u8;
        o[1] = (v >> 8) as u8;
        o[2] = v as u8;
    }

    // The 2- or 3-character tail (a remainder of 1 was rejected above). Its trailing bits are
    // dropped rather than checked for zero, which §3.5 leaves optional and which the
    // byte-at-a-time predecessor also did.
    let rest = &body[full * 4..];
    if rest.len() >= 2 {
        let a = DECODE[rest[0] as usize] as u32;
        let b = DECODE[rest[1] as usize] as u32;
        let c = if rest.len() == 3 { DECODE[rest[2] as usize] as u32 } else { 0 };
        if (a | b | c) & 0xC0 != 0 {
            return None;
        }
        tail[0] = ((a << 2) | (b >> 4)) as u8;
        if rest.len() == 3 {
            tail[1] = ((b << 4) | (c >> 2)) as u8;
        }
    }
    Some(out)
}

/// Not a base64 character. `0xFF` rather than any value ≤ 63, so that OR-ing several table
/// results and testing the `0xC0` mask detects an invalid byte anywhere in the group.
const BAD: u8 = 0xFF;

/// The reverse base64 alphabet (RFC 4648 §4, Table 1): `DECODE[c]` is the 6-bit value of
/// character `c`, or [`BAD`]. Built in `const` context from the same alphabet ranges the
/// byte-at-a-time decoder used to match on, so the table is derived from the specification
/// rather than transcribed from it — there is no 256-entry literal to mis-copy.
const DECODE: [u8; 256] = {
    let mut t = [BAD; 256];
    let mut i = 0usize;
    while i < 26 {
        t[b'A' as usize + i] = i as u8;
        t[b'a' as usize + i] = (i + 26) as u8;
        i += 1;
    }
    let mut d = 0usize;
    while d < 10 {
        t[b'0' as usize + d] = (d + 52) as u8;
        d += 1;
    }
    t[b'+' as usize] = 62;
    t[b'/' as usize] = 63;
    t
};

/// Parse a FLAC `PICTURE` block (RFC 9639 §8.8, Table 12) and return `(media type, image
/// bytes)`, both borrowed from `block`:
///
/// ```text
///   u32be picture_type | u32be mime_len | mime | u32be desc_len | desc
///   u32be width | u32be height | u32be depth | u32be colors
///   u32be data_len | data
/// ```
///
/// The picture type, description and the four informational dimensions are skipped: this
/// crate's sink vocabulary is `(mime, bytes)`, and §8.8 states applications "MUST NOT use"
/// the dimensions to decode the image. `None` on any truncation or non-ASCII media type
/// (§8.8: the media type "must be in printable ASCII characters 0x20-0x7E").
///
/// Duplicated rather than shared with `pf-flac`'s copy of this layout: `pf-ogg` is
/// codec-agnostic by design and depends on no codec crate (see the crate docs) — the whole
/// point of the container plugin. It is ~20 lines of table-driven field reads.
fn parse_picture_block(block: &[u8]) -> Option<(&str, &[u8])> {
    let be_u32 = |pos: usize| -> Option<usize> {
        let s = block.get(pos..)?.get(..4)?;
        Some(u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
    };
    let mut p = 4; // picture type (§8.8 Table 13) — not part of the sink vocabulary
    let mime_len = be_u32(p)?;
    p += 4;
    let mime = block.get(p..)?.get(..mime_len)?;
    if !mime.iter().all(|&b| (0x20..=0x7E).contains(&b)) {
        return None;
    }
    let mime = std::str::from_utf8(mime).ok()?;
    p += mime_len;
    let desc_len = be_u32(p)?;
    // description, then width/height/colour depth/indexed-colour count (4 × u32be).
    p = p.checked_add(4)?.checked_add(desc_len)?.checked_add(16)?;
    let data_len = be_u32(p)?;
    p += 4;
    let data = block.get(p..)?.get(..data_len)?;
    Some((mime, data))
}

#[cfg(test)]
// Tests build fixtures and record what the sink was handed; the parsers under test allocate
// nothing (spec: allocation discipline — tests are the sanctioned exception).
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    /// A `TagSink` that records what it was handed, so tests assert on the emitted stream.
    #[derive(Default)]
    struct Rec {
        text: Vec<(String, String)>,
        pictures: Vec<(String, Vec<u8>)>,
    }

    impl TagSink for Rec {
        fn text(&mut self, key: &str, value: &str) {
            self.text.push((key.to_string(), value.to_string()));
        }
        fn picture(&mut self, mime: &str, data: &[u8]) {
            self.pictures.push((mime.to_string(), data.to_vec()));
        }
    }

    /// Build a comment body (Vorbis I §5.2.2.1) from a vendor string and entries.
    fn body(vendor: &str, entries: &[&str]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        b.extend_from_slice(vendor.as_bytes());
        b.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for e in entries {
            b.extend_from_slice(&(e.len() as u32).to_le_bytes());
            b.extend_from_slice(e.as_bytes());
        }
        b
    }

    /// A FLAC PICTURE block (RFC 9639 §8.8 Table 12) around `data`.
    fn picture_block(mime: &str, desc: &str, data: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&3u32.to_be_bytes()); // type 3: front cover
        b.extend_from_slice(&(mime.len() as u32).to_be_bytes());
        b.extend_from_slice(mime.as_bytes());
        b.extend_from_slice(&(desc.len() as u32).to_be_bytes());
        b.extend_from_slice(desc.as_bytes());
        b.extend_from_slice(&[0u8; 16]); // width, height, depth, colours
        b.extend_from_slice(&(data.len() as u32).to_be_bytes());
        b.extend_from_slice(data);
        b
    }

    /// Reference base64 encoder (RFC 4648 §4) for the tests — deliberately written the
    /// other way round from the decoder so the two do not share a bug.
    fn base64_encode(data: &[u8]) -> String {
        const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut s = String::new();
        for chunk in data.chunks(3) {
            let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
            let n = u32::from_be_bytes([0, b[0], b[1], b[2]]);
            s.push(A[(n >> 18) as usize & 63] as char);
            s.push(A[(n >> 12) as usize & 63] as char);
            s.push(if chunk.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
            s.push(if chunk.len() > 2 { A[n as usize & 63] as char } else { '=' });
        }
        s
    }

    fn arena() -> Arena {
        Arena::new(Arena::DEFAULT_CHUNK)
    }

    #[test]
    fn parses_entries_uppercases_keys_and_keeps_repeats() {
        let b = body(
            "reference libFLAC",
            &[
                "TITLE=Set Theory",
                "artist=Carbon Based Lifeforms",
                "ARTIST=Second Artist",
                "REPLAYGAIN_TRACK_GAIN=-7.18 dB",
                "Comment=hi=there",
            ],
        );
        let mut rec = Rec::default();
        parse_comment_body(&b, &arena(), &mut rec);
        assert_eq!(
            rec.text,
            vec![
                ("TITLE".to_string(), "Set Theory".to_string()),
                ("ARTIST".to_string(), "Carbon Based Lifeforms".to_string()),
                ("ARTIST".to_string(), "Second Artist".to_string()),
                ("REPLAYGAIN_TRACK_GAIN".to_string(), "-7.18 dB".to_string()),
                // Only the first '=' splits, so values may contain '='.
                ("COMMENT".to_string(), "hi=there".to_string()),
            ]
        );
        assert!(rec.pictures.is_empty());
    }

    #[test]
    fn skips_malformed_entries_and_continues() {
        // No '=', an empty field name, an illegal field-name byte (0x7F), invalid UTF-8 —
        // each entry is dropped, and the good entries around them still arrive. Built from
        // raw bytes because one entry is deliberately not UTF-8.
        let entries: [&[u8]; 6] = [
            b"TITLE=ok",
            b"novalue",
            b"=novalue",
            b"BAD\x7FKEY=x",
            b"NAME=\xFF\xFE",
            b"ALBUM=fine",
        ];
        let mut b = Vec::new();
        b.extend_from_slice(&1u32.to_le_bytes());
        b.push(b'v');
        b.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for e in entries {
            b.extend_from_slice(&(e.len() as u32).to_le_bytes());
            b.extend_from_slice(e);
        }

        let mut rec = Rec::default();
        parse_comment_body(&b, &arena(), &mut rec);
        assert_eq!(
            rec.text,
            vec![("TITLE".to_string(), "ok".to_string()), ("ALBUM".to_string(), "fine".to_string())]
        );
    }

    #[test]
    fn tilde_field_name_accepted_rfc9639_range() {
        // Vorbis I stops at 0x7D, RFC 9639 §8.6 allows 0x7E — we accept the union.
        let mut rec = Rec::default();
        parse_comment_body(&body("v", &["A~B=x"]), &arena(), &mut rec);
        assert_eq!(rec.text, vec![("A~B".to_string(), "x".to_string())]);
    }

    #[test]
    fn picture_round_trips_through_base64() {
        let jpeg = b"\xFF\xD8\xFF\xE0jpeg-ish bytes";
        let block = picture_block("image/jpeg", "front", jpeg);
        let entry = format!("METADATA_BLOCK_PICTURE={}", base64_encode(&block));
        let mut rec = Rec::default();
        parse_comment_body(&body("v", &["TITLE=x", &entry]), &arena(), &mut rec);
        assert_eq!(rec.text, vec![("TITLE".to_string(), "x".to_string())]);
        assert_eq!(rec.pictures.len(), 1);
        assert_eq!(rec.pictures[0].0, "image/jpeg");
        assert_eq!(rec.pictures[0].1, jpeg);
    }

    /// The three padding shapes RFC 4648 §4 can produce, driven through a real picture so
    /// each is exercised end to end rather than as a base64 unit test.
    #[test]
    fn picture_covers_all_base64_pad_variants() {
        let mut seen = Vec::new();
        for extra in 0..3usize {
            // The block length modulo 3 selects the padding; grow the description to move it.
            let data = b"IMG";
            let block = picture_block("image/png", &"d".repeat(extra), data);
            let b64 = base64_encode(&block);
            let pad: String = b64.chars().rev().take_while(|&c| c == '=').collect();
            let want_pad = match block.len() % 3 {
                0 => "",
                1 => "==",
                _ => "=",
            };
            assert_eq!(pad, want_pad, "block len {} encodes with pad {want_pad:?}", block.len());
            seen.push(pad);

            let entry = format!("METADATA_BLOCK_PICTURE={b64}");
            let mut rec = Rec::default();
            parse_comment_body(&body("v", &[&entry]), &arena(), &mut rec);
            assert_eq!(rec.pictures.len(), 1, "pad {want_pad:?} must decode");
            assert_eq!(rec.pictures[0].1, data);

            // The same payload with its padding stripped is still accepted (§3.2).
            let entry = format!("METADATA_BLOCK_PICTURE={}", b64.trim_end_matches('='));
            let mut rec = Rec::default();
            parse_comment_body(&body("v", &[&entry]), &arena(), &mut rec);
            assert_eq!(rec.pictures.len(), 1, "unpadded pad {want_pad:?} must decode");
            assert_eq!(rec.pictures[0].1, data);
        }
        seen.sort_unstable();
        assert_eq!(seen, ["", "=", "=="], "all three RFC 4648 §4 pad shapes must be covered");
    }

    #[test]
    fn invalid_base64_skips_the_entry_and_parsing_continues() {
        let good = base64_encode(&picture_block("image/png", "", b"IMG"));
        for bad in [
            format!("{good}extra=chars"),      // '=' in the middle
            format!("{}!", &good[..good.len() - 1]), // non-alphabet byte
            good[..good.len() - 3].to_string(), // remainder of 1 character
            "A".to_string(),                    // remainder of 1 character, whole input
            format!("{good}===="),              // over-padded
        ] {
            let entry = format!("METADATA_BLOCK_PICTURE={bad}");
            let mut rec = Rec::default();
            parse_comment_body(&body("v", &["TITLE=before", &entry, "ALBUM=after"]), &arena(), &mut rec);
            assert!(rec.pictures.is_empty(), "must not decode {bad:?}");
            assert_eq!(
                rec.text,
                vec![
                    ("TITLE".to_string(), "before".to_string()),
                    ("ALBUM".to_string(), "after".to_string())
                ],
                "the walk must continue past {bad:?}"
            );
        }
    }

    #[test]
    fn base64_decodes_rfc4648_test_vectors() {
        // RFC 4648 §10 test vectors.
        let a = arena();
        for (plain, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64_decode(encoded.as_bytes(), &a), Some(plain.as_bytes()), "{encoded}");
        }
    }

    /// The table-driven decoder must agree, byte for byte and rejection for rejection, with
    /// the textbook bit-at-a-time definition it replaced — the same shape of proof
    /// `crc::tests::table_step_matches_bitwise` gives the CRC table.
    ///
    /// Inputs are drawn to hit the cases that actually differ between the two: every length
    /// modulo 4, padding in legal and illegal positions, and bytes just outside the alphabet
    /// (`.`, `-`, `\n`, 0x00, 0xFF) which a naive `match` and a 256-entry table can disagree
    /// about.
    #[test]
    fn table_decode_matches_bitwise_reference() {
        /// The predecessor, verbatim: strip ≤2 trailing pads, reject `=` elsewhere, then
        /// accumulate 6 bits at a time.
        fn reference(input: &[u8]) -> Option<Vec<u8>> {
            let mut pad = 0;
            let mut n = input.len();
            while pad < 2 && n > 0 && input[n - 1] == b'=' {
                n -= 1;
                pad += 1;
            }
            let body = &input[..n];
            if body.contains(&b'=') {
                return None;
            }
            let rem = body.len() % 4;
            if rem == 1 || (pad > 0 && (rem + pad != 4)) {
                return None;
            }
            let mut out = Vec::new();
            let (mut acc, mut bits) = (0u32, 0u32);
            for &c in body {
                let v = match c {
                    b'A'..=b'Z' => c - b'A',
                    b'a'..=b'z' => c - b'a' + 26,
                    b'0'..=b'9' => c - b'0' + 52,
                    b'+' => 62,
                    b'/' => 63,
                    _ => return None,
                };
                acc = (acc << 6) | v as u32;
                bits += 6;
                if bits >= 8 {
                    bits -= 8;
                    out.push((acc >> bits) as u8);
                }
            }
            Some(out)
        }

        let alphabet: Vec<u8> =
            (b'A'..=b'Z').chain(b'a'..=b'z').chain(b'0'..=b'9').chain(*b"+/").collect();
        // A cheap deterministic PRNG (xorshift64*) keeps the test dependency-free and its
        // failures reproducible.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let odd = [b'=', b'.', b'-', b'\n', b' ', 0x00, 0xFF];

        let a = arena();
        for len in 0..40usize {
            for trial in 0..64 {
                let mut v: Vec<u8> = (0..len)
                    .map(|_| alphabet[(next() % alphabet.len() as u64) as usize])
                    .collect();
                // Half the trials splice in a byte from outside the alphabet, at a random
                // position — including the tail, where `=` is legal.
                if trial % 2 == 1 && len > 0 {
                    let at = (next() % len as u64) as usize;
                    v[at] = odd[(next() % odd.len() as u64) as usize];
                }
                let got = base64_decode(&v, &a).map(<[u8]>::to_vec);
                assert_eq!(got, reference(&v), "input {v:?}");
            }
        }
    }

    #[test]
    fn truncated_picture_block_is_dropped() {
        let block = picture_block("image/png", "d", b"IMGDATA");
        for cut in 0..block.len() {
            let entry = format!("METADATA_BLOCK_PICTURE={}", base64_encode(&block[..cut]));
            let mut rec = Rec::default();
            parse_comment_body(&body("v", &[&entry]), &arena(), &mut rec);
            assert!(rec.pictures.is_empty(), "truncated to {cut} bytes must not emit");
        }
    }

    #[test]
    fn truncations_never_panic() {
        let block = picture_block("image/jpeg", "front", b"IMG");
        let entry = format!("METADATA_BLOCK_PICTURE={}", base64_encode(&block));
        let full = body("reference", &["TITLE=Set Theory", "ARTIST=x", &entry]);
        for n in 0..full.len() {
            let mut rec = Rec::default();
            parse_comment_body(&full[..n], &arena(), &mut rec);
        }
        // A lying vendor length / comment count must not read past the body either.
        let mut lying = full.clone();
        lying[0..4].copy_from_slice(&u32::MAX.to_le_bytes());
        parse_comment_body(&lying, &arena(), &mut Rec::default());
        let mut lying = full.clone();
        let count_at = 4 + 9; // vendor "reference"
        lying[count_at..count_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        parse_comment_body(&lying, &arena(), &mut Rec::default());
    }

    #[test]
    fn long_field_name_uses_the_arena_not_the_stack() {
        let key = "k".repeat(KEY_STACK_MAX + 17);
        let entry = format!("{key}=v");
        let mut rec = Rec::default();
        parse_comment_body(&body("v", &[&entry]), &arena(), &mut rec);
        assert_eq!(rec.text, vec![(key.to_ascii_uppercase(), "v".to_string())]);
    }

    #[test]
    fn opus_tags_wrapper_needs_its_magic() {
        let mut packet = b"OpusTags".to_vec();
        packet.extend_from_slice(&body("libopus 1.4", &["TITLE=Opus"]));
        let mut rec = Rec::default();
        assert!(parse_opus_tags(&packet, &arena(), &mut rec));
        assert_eq!(rec.text, vec![("TITLE".to_string(), "Opus".to_string())]);

        let mut rec = Rec::default();
        assert!(!parse_opus_tags(b"OpusHead\0\0\0", &arena(), &mut rec));
        assert!(rec.text.is_empty());
        assert!(!parse_opus_tags(b"Opus", &arena(), &mut Rec::default()));
    }

    #[test]
    fn vorbis_comment_wrapper_needs_its_header_and_ignores_the_framing_bit() {
        let mut packet = b"\x03vorbis".to_vec();
        packet.extend_from_slice(&body("Xiph.Org libVorbis", &["TITLE=Vorbis"]));
        packet.push(0x01); // framing bit (Vorbis I §5.2.2.1) — ignored
        let mut rec = Rec::default();
        assert!(parse_vorbis_comments(&packet, &arena(), &mut rec));
        assert_eq!(rec.text, vec![("TITLE".to_string(), "Vorbis".to_string())]);

        // Even with the framing bit cleared (illegal, per the spec) the tags still parse.
        let mut zeroed = packet.clone();
        *zeroed.last_mut().unwrap() = 0x00;
        let mut rec = Rec::default();
        assert!(parse_vorbis_comments(&zeroed, &arena(), &mut rec));
        assert_eq!(rec.text.len(), 1);

        // The identification header (0x01) is not the comment header.
        assert!(!parse_vorbis_comments(b"\x01vorbis....", &arena(), &mut Rec::default()));
    }
}
