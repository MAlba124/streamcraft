//! FLAC metadata-block tag extraction: `VORBIS_COMMENT` (§8.6) and `PICTURE` (§8.7).
//!
//! The metadata-block chain is byte-aligned, so this parses the raw header bytes directly and is
//! independent of the bit-level frame decoder. Both the one-shot [`FlacDecoder`](crate::FlacDecoder)
//! and the streaming [`StreamDecoder`](crate::StreamDecoder) fill a [`FlacTags`] from it, and the
//! [`FlacDec`](crate::FlacDec) element turns that into a `profluens_core::event::TagList` it emits.
//! Parsing is best-effort: malformed or truncated metadata simply stops the walk, never errors.

/// Tags parsed from a FLAC stream's metadata blocks.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct FlacTags {
    /// Vorbis comments as `(uppercased field name, value)` — e.g. `("TITLE", "Enough")`. A field
    /// may repeat (two `ARTIST`s), and order is preserved.
    pub comments: Vec<(String, String)>,
    /// Attached pictures as `(MIME type, image bytes)`, e.g. `("image/jpeg", <jpeg>)`.
    pub pictures: Vec<(String, Vec<u8>)>,
}

impl FlacTags {
    /// No comments and no pictures.
    pub fn is_empty(&self) -> bool {
        self.comments.is_empty() && self.pictures.is_empty()
    }

    /// Total number of comments plus pictures.
    pub fn len(&self) -> usize {
        self.comments.len() + self.pictures.len()
    }
}

/// Walk a FLAC stream's metadata-block chain (starting at the `fLaC` marker) and extract its tags.
// One-time: a stream's tags are parsed once, at header time — not the per-frame decode path (spec:
// allocation discipline — header parsing is the sanctioned one-time-allocation exception).
#[allow(clippy::disallowed_methods)]
pub fn parse(bytes: &[u8]) -> FlacTags {
    let mut tags = FlacTags::default();
    if bytes.len() < 4 || &bytes[0..4] != b"fLaC" {
        return tags;
    }
    // Each block: 1 header byte `[last:1 | type:7]`, a 3-byte big-endian length, then the body.
    let mut pos = 4;
    while pos + 4 <= bytes.len() {
        let header = bytes[pos];
        let last = header & 0x80 != 0;
        let block_type = header & 0x7f;
        let len = u32::from_be_bytes([0, bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]]) as usize;
        let start = pos + 4;
        let Some(block) = bytes.get(start..start + len) else { break };
        match block_type {
            4 => parse_vorbis_comment(block, &mut tags),
            6 => parse_picture(block, &mut tags),
            _ => {}
        }
        pos = start + len;
        if last {
            break;
        }
    }
    tags
}

/// `VORBIS_COMMENT` body (§8.6): a little-endian-length-prefixed vendor string (skipped), then a
/// count and that many length-prefixed UTF-8 `FIELD=value` entries.
#[allow(clippy::disallowed_methods)] // one-time — see `parse`
fn parse_vorbis_comment(block: &[u8], tags: &mut FlacTags) {
    let le = |b: &[u8], p: usize| {
        b.get(p..p + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
    };
    let mut p = 0;
    let Some(vendor_len) = le(block, p) else { return };
    p += 4 + vendor_len;
    let Some(count) = le(block, p) else { return };
    p += 4;
    for _ in 0..count {
        let Some(clen) = le(block, p) else { break };
        p += 4;
        let Some(s) = block.get(p..p + clen) else { break };
        p += clen;
        if let Ok(s) = std::str::from_utf8(s) {
            if let Some((key, value)) = s.split_once('=') {
                tags.comments.push((key.to_ascii_uppercase(), value.to_string()));
            }
        }
    }
}

/// `PICTURE` body (§8.7): picture type (4 BE), MIME (len + bytes), description (len + bytes),
/// width/height/depth/colors (4 BE each), then the image data (len + bytes).
#[allow(clippy::disallowed_methods)] // one-time — see `parse`
fn parse_picture(block: &[u8], tags: &mut FlacTags) {
    let be = |b: &[u8], p: usize| {
        b.get(p..p + 4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
    };
    let mut p = 4; // skip picture type
    let Some(mime_len) = be(block, p) else { return };
    p += 4;
    let Some(mime) = block.get(p..p + mime_len).and_then(|s| std::str::from_utf8(s).ok()) else {
        return;
    };
    let mime = mime.to_string();
    p += mime_len;
    let Some(desc_len) = be(block, p) else { return };
    p += 4 + desc_len; // skip description
    p += 16; // width, height, colour depth, indexed-colour count (4 × 4 bytes)
    let Some(data_len) = be(block, p) else { return };
    p += 4;
    let Some(data) = block.get(p..p + data_len) else { return };
    tags.pictures.push((mime, data.to_vec()));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal FLAC header: `fLaC`, a (skipped) STREAMINFO, then a VORBIS_COMMENT block.
    fn flac_with_comments(comments: &[&str]) -> Vec<u8> {
        let mut out = b"fLaC".to_vec();
        // STREAMINFO block (type 0, len 34, zeroed body — content irrelevant to tag parsing).
        out.push(0);
        out.extend_from_slice(&[0, 0, 34]);
        out.extend_from_slice(&[0u8; 34]);
        // VORBIS_COMMENT block (type 4), last-block flag set.
        let mut body = Vec::new();
        let vendor = b"ref";
        body.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        body.extend_from_slice(vendor);
        body.extend_from_slice(&(comments.len() as u32).to_le_bytes());
        for c in comments {
            body.extend_from_slice(&(c.len() as u32).to_le_bytes());
            body.extend_from_slice(c.as_bytes());
        }
        out.push(0x80 | 4);
        let len = body.len() as u32;
        out.extend_from_slice(&len.to_be_bytes()[1..]);
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn parses_vorbis_comments() {
        let flac = flac_with_comments(&["TITLE=Set Theory", "ARTIST=Carbon Based Lifeforms", "DATE=2015"]);
        let tags = parse(&flac);
        assert_eq!(
            tags.comments,
            vec![
                ("TITLE".into(), "Set Theory".into()),
                ("ARTIST".into(), "Carbon Based Lifeforms".into()),
                ("DATE".into(), "2015".into()),
            ]
        );
        assert!(tags.pictures.is_empty());
        assert_eq!(tags.len(), 3);
    }

    #[test]
    fn lowercase_keys_are_uppercased_values_kept() {
        let tags = parse(&flac_with_comments(&["title=x", "Comment=hi=there"]));
        // Key uppercased; only the first '=' splits, so values may contain '='.
        assert_eq!(tags.comments[0], ("TITLE".into(), "x".into()));
        assert_eq!(tags.comments[1], ("COMMENT".into(), "hi=there".into()));
    }

    #[test]
    fn non_flac_or_empty_is_empty() {
        assert!(parse(b"not a flac file").is_empty());
        assert!(parse(&[]).is_empty());
        assert!(parse(b"fLaC").is_empty()); // marker only, no blocks
    }

    /// A PICTURE block body (§8.7): type, MIME, description, dimensions, then the image bytes.
    fn picture_block_body(mime: &str, data: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&3u32.to_be_bytes()); // picture type: front cover
        b.extend_from_slice(&(mime.len() as u32).to_be_bytes());
        b.extend_from_slice(mime.as_bytes());
        b.extend_from_slice(&(3u32).to_be_bytes()); // description length
        b.extend_from_slice(b"pic"); // description
        b.extend_from_slice(&[0u8; 16]); // width, height, colour depth, indexed colours
        b.extend_from_slice(&(data.len() as u32).to_be_bytes());
        b.extend_from_slice(data);
        b
    }

    #[test]
    fn parses_pictures() {
        // fLaC + STREAMINFO + PICTURE(last).
        let mut flac = b"fLaC".to_vec();
        flac.push(0);
        flac.extend_from_slice(&[0, 0, 34]);
        flac.extend_from_slice(&[0u8; 34]);
        let body = picture_block_body("image/jpeg", b"\xFF\xD8\xFF\xE0jpeg-ish bytes");
        flac.push(0x80 | 6); // last-block, PICTURE
        flac.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
        flac.extend_from_slice(&body);

        let tags = parse(&flac);
        assert_eq!(tags.pictures.len(), 1);
        assert_eq!(tags.pictures[0].0, "image/jpeg");
        assert_eq!(tags.pictures[0].1, b"\xFF\xD8\xFF\xE0jpeg-ish bytes");
        assert!(tags.comments.is_empty());
    }

    #[test]
    fn parses_comments_and_a_picture_together() {
        // fLaC + STREAMINFO(not-last) + VORBIS_COMMENT(not-last) + PICTURE(last).
        let mut flac = b"fLaC".to_vec();
        flac.push(0);
        flac.extend_from_slice(&[0, 0, 34]);
        flac.extend_from_slice(&[0u8; 34]);
        // VORBIS_COMMENT (not last)
        let mut vc = Vec::new();
        vc.extend_from_slice(&3u32.to_le_bytes());
        vc.extend_from_slice(b"ref");
        vc.extend_from_slice(&1u32.to_le_bytes());
        vc.extend_from_slice(&("TITLE=X".len() as u32).to_le_bytes());
        vc.extend_from_slice(b"TITLE=X");
        flac.push(4);
        flac.extend_from_slice(&(vc.len() as u32).to_be_bytes()[1..]);
        flac.extend_from_slice(&vc);
        // PICTURE (last)
        let pic = picture_block_body("image/png", b"\x89PNG...");
        flac.push(0x80 | 6);
        flac.extend_from_slice(&(pic.len() as u32).to_be_bytes()[1..]);
        flac.extend_from_slice(&pic);

        let tags = parse(&flac);
        assert_eq!(tags.comments, vec![("TITLE".to_string(), "X".to_string())]);
        assert_eq!(tags.pictures.len(), 1);
        assert_eq!(tags.pictures[0].0, "image/png");
        assert_eq!(tags.pictures[0].1, b"\x89PNG...");
    }
}
