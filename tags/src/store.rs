//! Where a scanned file's tags live for the duration of the callback: an arena for text,
//! refcounted [`Memory`] views for pictures.
//!
//! The parsers in this crate never build owned strings. They push into a [`TagSink`]
//! (core's streaming tag face), and [`ArenaSink`] — the one sink the engine installs —
//! copies each key/value into the scanner's bump [`Arena`] and each picture into either a
//! zero-copy slice of the read buffer or, when the bytes do not live there, one
//! right-sized pool buffer.
//!
//! The arena is reset after every file, which is exactly why [`Tags`] is lifetime-bound:
//! the borrow checker will not let a caller keep one past the callback. A caller that
//! wants to keep the tags calls [`Tags::to_tag_list`] and gets core's owned
//! [`TagList`](profluens_core::event::TagList) — the one copy, on request.

use profluens_core::event::{TagList, TagListSink, TagSink};
use profluens_core::memory::{Arena, Memory, Pool};

/// An attached picture (cover art). The bytes are a refcounted view, not a copy, whenever
/// they were read into a buffer this scanner owns — forwarding cover art through the graph
/// then costs a refcount bump (spec: Memory).
pub struct PictureRef<'a> {
    pub mime: &'a str,
    pub data: Memory,
}

/// One file's tags: text in the scanner's arena, pictures in [`Memory`].
///
/// Text keys are the canonical uppercase Vorbis-comment vocabulary core's
/// [`TagSink`] documents (`TITLE`, `ARTIST`, `ALBUM`, `ALBUMARTIST`, `TRACKNUMBER`,
/// `DISCNUMBER`, `DATE`, `GENRE`, `COMMENT`, `REPLAYGAIN_TRACK_GAIN`, …), and a key may
/// repeat, in file order.
pub struct Tags<'a> {
    text: Vec<(&'a str, &'a str), &'a Arena>,
    pictures: Vec<PictureRef<'a>, &'a Arena>,
}

impl<'a> Tags<'a> {
    /// An empty set of tags backed by `arena`. Allocates nothing until the first push
    /// (`Vec::new_in` does not touch its allocator).
    pub(crate) fn new_in(arena: &'a Arena) -> Self {
        Self { text: Vec::new_in(arena), pictures: Vec::new_in(arena) }
    }

    /// First value for `key`, case-insensitively.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.text.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)).map(|(_, v)| *v)
    }

    /// All `(key, value)` pairs in file order; a key may repeat.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.text.iter().map(|(k, v)| (*k, *v))
    }

    /// The attached pictures.
    pub fn pictures(&self) -> &[PictureRef<'a>] {
        &self.pictures
    }

    /// Number of text tags plus pictures.
    pub fn len(&self) -> usize {
        self.text.len() + self.pictures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.pictures.is_empty()
    }

    /// Materialise core's owned [`TagList`] — the copy a caller pays only when it wants to
    /// keep the tags past the callback. Driven through [`TagListSink`], so the picture
    /// bytes land in `pool` (right-sized, never pinning a slot) exactly as a demuxer's
    /// would.
    pub fn to_tag_list(&self, pool: &Pool) -> TagList {
        let mut sink = TagListSink::new(pool);
        for (k, v) in self.iter() {
            sink.text(k, v);
        }
        for p in self.pictures() {
            sink.picture(p.mime, p.data.data());
        }
        sink.finish()
    }
}

/// The engine's [`TagSink`]: interns text into the arena, keeps pictures zero-copy.
pub(crate) struct ArenaSink<'a> {
    arena: &'a Arena,
    pool: &'a Pool,
    /// The buffer the parser is reading from. A picture whose bytes lie inside it becomes a
    /// [`Memory::slice`] — a refcount bump, no copy. `None` (or bytes from anywhere else)
    /// falls back to a right-sized pool copy.
    src: Option<&'a Memory>,
    tags: Tags<'a>,
}

impl<'a> ArenaSink<'a> {
    pub(crate) fn new(arena: &'a Arena, pool: &'a Pool, src: Option<&'a Memory>) -> Self {
        Self { arena, pool, src, tags: Tags::new_in(arena) }
    }

    pub(crate) fn finish(self) -> Tags<'a> {
        self.tags
    }
}

impl<'a> TagSink for ArenaSink<'a> {
    fn text(&mut self, key: &str, value: &str) {
        // `&'a Arena` is `Copy`, so the interned strings keep the arena's lifetime rather
        // than the `&mut self` reborrow's.
        let arena: &'a Arena = self.arena;
        self.tags.text.push((intern(arena, key), intern(arena, value)));
    }

    fn picture(&mut self, mime: &str, data: &[u8]) {
        let arena: &'a Arena = self.arena;
        let mime = intern(arena, mime);
        let mem = match self.src.and_then(|src| view_of(src, data)) {
            // The bytes are inside the buffer we read them into: a sub-view, zero copies
            // (spec: Memory — "sub-buffer slicing is free").
            Some(view) => view,
            // They are not (an owning parser handed us its own storage): one right-sized
            // pool copy, the same `acquire_exact` policy `TagListSink::picture` uses, so a
            // multi-MiB cover never pins a pool slot.
            None => {
                let mut m = self.pool.acquire_exact(data.len());
                // A recycled buffer arrives holding stale bytes and may be larger than
                // asked: write the exact prefix, then trim to it.
                m.as_mut_full()[..data.len()].copy_from_slice(data);
                m.set_len(data.len());
                m
            }
        };
        self.tags.pictures.push(PictureRef { mime, data: mem });
    }
}

/// Copy `s` into `arena` and return it as a `&str` with the arena's lifetime.
///
/// The re-validation is free of `unsafe` and cannot fail — the bytes came from a `&str` —
/// so the fallback is unreachable, not a silent data loss.
pub(crate) fn intern<'a>(arena: &'a Arena, s: &str) -> &'a str {
    let region = arena.alloc_bytes(s.len());
    region.copy_from_slice(s.as_bytes());
    std::str::from_utf8(region).unwrap_or("")
}

/// A zero-copy [`Memory`] view of `data`, if `data` points inside `src`'s used bytes.
///
/// A plain address-range test: the parsers hand out sub-slices of the buffer they were
/// given, so "is this ours" is `src.data().as_ptr() <= data && data + len <= end`. Casting
/// the pointers to `usize` keeps the whole check safe code.
fn view_of(src: &Memory, data: &[u8]) -> Option<Memory> {
    let base = src.data();
    if data.is_empty() || base.is_empty() {
        return None;
    }
    let lo = base.as_ptr() as usize;
    let hi = lo.checked_add(base.len())?;
    let start = data.as_ptr() as usize;
    let end = start.checked_add(data.len())?;
    if start < lo || end > hi {
        return None;
    }
    src.slice(start - lo, data.len())
}

/// Decode `bytes` as text into `scratch`, returning a `&str` view of it.
///
/// Container text fields that predate Unicode (a RIFF `LIST`/`INFO` `ZSTR`, an ID3v2
/// ISO-8859-1 frame) are "UTF-8 if it happens to be valid, otherwise Latin-1" in practice —
/// no field records which. Valid UTF-8 passes through untouched and unbounded; the Latin-1
/// fallback transcodes into the caller's fixed `scratch` (every byte ≥ 0x80 becomes two),
/// which bounds that path at `scratch.len() / 2` source bytes. Rare and short by
/// construction: these are titles and comments, not payload.
pub(crate) fn decode_text<'s>(bytes: &'s [u8], scratch: &'s mut [u8]) -> &'s str {
    if let Ok(s) = std::str::from_utf8(bytes) {
        return s;
    }
    // ISO-8859-1 → UTF-8: code points U+0000..U+00FF, so one or two bytes each
    // (Unicode 15.0 §3.9, Table 3-6).
    let mut n = 0;
    for &b in bytes {
        if b < 0x80 {
            if n + 1 > scratch.len() {
                break;
            }
            scratch[n] = b;
            n += 1;
        } else {
            if n + 2 > scratch.len() {
                break;
            }
            scratch[n] = 0xC0 | (b >> 6);
            scratch[n + 1] = 0x80 | (b & 0x3F);
            n += 2;
        }
    }
    std::str::from_utf8(&scratch[..n]).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interned_text_survives_the_source() {
        let arena = Arena::default();
        let pool = Pool::bounded(4096, 4);
        let mut sink = ArenaSink::new(&arena, &pool, None);
        {
            let owned = String::from("Carbon Based Lifeforms");
            sink.text("ARTIST", &owned);
        }
        let tags = sink.finish();
        assert_eq!(tags.get("artist"), Some("Carbon Based Lifeforms"));
        assert_eq!(tags.get("TITLE"), None);
        assert_eq!(tags.len(), 1);
    }

    #[test]
    fn picture_inside_the_read_buffer_is_zero_copy() {
        let arena = Arena::default();
        let pool = Pool::bounded(4096, 4);
        let mut src = pool.acquire_exact(4096);
        src.as_mut_full()[..8].copy_from_slice(b"\x89PNG\r\n\x1a\n");
        src.set_len(8);
        let base = src.data().as_ptr() as usize;

        let mut sink = ArenaSink::new(&arena, &pool, Some(&src));
        let inner = &src.data()[2..6];
        sink.picture("image/png", inner);
        let tags = sink.finish();

        let p = &tags.pictures()[0];
        assert_eq!(p.mime, "image/png");
        assert_eq!(p.data.data(), b"NG\r\n");
        // Same backing bytes, not a copy: the view's address is inside the source buffer.
        assert_eq!(p.data.data().as_ptr() as usize, base + 2);
    }

    #[test]
    fn picture_outside_the_read_buffer_is_copied() {
        let arena = Arena::default();
        let pool = Pool::bounded(4096, 4);
        let mut src = pool.acquire_exact(4096);
        src.set_len(8);
        let elsewhere = vec![7u8; 300];

        let mut sink = ArenaSink::new(&arena, &pool, Some(&src));
        sink.picture("image/jpeg", &elsewhere);
        let tags = sink.finish();

        let p = &tags.pictures()[0];
        assert_eq!(p.data.len(), 300);
        assert_eq!(p.data.data(), &elsewhere[..]);
        assert_ne!(p.data.data().as_ptr() as usize, elsewhere.as_ptr() as usize);
    }

    #[test]
    fn to_tag_list_materialises_an_owned_copy() {
        let arena = Arena::default();
        let pool = Pool::bounded(4096, 8);
        let mut sink = ArenaSink::new(&arena, &pool, None);
        sink.text("TITLE", "Enough");
        sink.picture("image/png", b"\x89PNG");
        let list = sink.finish().to_tag_list(&pool);
        assert_eq!(list.get("TITLE"), Some("Enough"));
        assert_eq!(list.pictures()[0].data.data(), b"\x89PNG");
    }

    #[test]
    fn latin1_text_is_transcoded_utf8_passes_through() {
        let mut scratch = [0u8; 64];
        assert_eq!(decode_text("Björk".as_bytes(), &mut scratch), "Björk");
        // 0xF6 is Latin-1 'ö'; invalid alone as UTF-8.
        assert_eq!(decode_text(b"Bj\xf6rk", &mut scratch), "Björk");
        // A scratch too small truncates rather than panicking.
        let mut tiny = [0u8; 3];
        assert_eq!(decode_text(b"Bj\xf6rk", &mut tiny), "Bj");
    }
}
