//! iTunes-style (`ilst`) metadata and no-decode duration/format props — pure `&[u8]` parsing.
//!
//! A tag scanner wants three things out of an MP4 and none of them involve decoding: *where*
//! the `moov` box is (it may sit before or after a multi-gigabyte `mdat`), the iTunes metadata
//! inside it, and the duration/rate/channels a listing view shows. This module answers all
//! three from borrowed slices — no IO, no element, no pipeline — and, unlike the rest of the
//! crate (which builds owned sample tables once per stream), with **no heap allocation at
//! all**: text is emitted as a borrow of the input, formatted numbers go to a stack buffer,
//! and the one variable-length key (a `----` freeform name) goes to the caller's [`Arena`].
//!
//! ## The three entry points
//! - [`locate_moov`] — walk the top-level boxes of a bounded *prefix* read (§4.2) and report
//!   whether `moov` is inside it, past its end, or absent.
//! - [`parse_moov_tags`] — `moov` → `udta` → `meta` → `ilst` (§8.10.1, §8.11.1; the item
//!   scheme itself is Apple's), pushing canonical Vorbis-comment-style keys into a
//!   [`TagSink`].
//! - [`props_from_moov`] — duration from the sound track's `mdhd` (§8.4.2) with an `mvhd`
//!   fallback (§8.2.2), plus rate/channels off the audio sample entry (§12.2.3).
//!
//! Everything is best-effort over untrusted bytes (spec: "a crash on bad input is a P0"): an
//! unknown item, an unknown data type, a truncated box or a non-UTF-8 string is skipped
//! silently, never a panic and never an error the caller has to handle — a tag scanner over a
//! directory of half-downloaded files must simply keep going.
//!
//! ## Spec provenance (`mp4/spec/NOTES.md`)
//! Box grammar and the `moov` interior: **ISO/IEC 14496-12** (6th ed., 2022), sections cited
//! inline. The metadata *item* scheme (`ilst`, the `data` atom's type indicator + locale, the
//! `----`/`mean`/`name` freeform triple, the well-known type numbers) is **Apple's**, from the
//! published *QuickTime File Format Specification*, "Metadata" chapter — cited inline as
//! (QTFF Metadata) since it has no ISO clause numbers. Clean-room: written from the standards
//! only, no consultation of any existing tag-reading implementation.

use core::fmt::Write as _;

use profluens_core::event::TagSink;
use profluens_core::memory::Arena;

use crate::boxes::{self, boxtype, FourCc};

// =====================================================================================
// Box types this module walks. They live here rather than in `boxes::boxtype` (which lists
// what the sample-table resolver needs) because they are metadata-only — a demuxer never
// touches them.
// =====================================================================================

/// `udta` UserDataBox (§8.10.1) — where a movie parks non-normative extras, including the
/// iTunes `meta`.
const UDTA: FourCc = boxes::fourcc(b"udta");
/// `meta` MetaBox (§8.11.1) — **a FullBox**; see [`ilst_children`].
const META: FourCc = boxes::fourcc(b"meta");
/// `hdlr` HandlerBox (§8.4.3) — the `soun`/`vide` discriminator inside `mdia` (and the
/// `mdir` metadata-handler declaration inside `meta`, which we do not need to read).
const HDLR: FourCc = boxes::fourcc(b"hdlr");
/// `ilst` metadata item list (QTFF Metadata) — the container of the tag items.
const ILST: FourCc = boxes::fourcc(b"ilst");
/// `data` metadata item data atom (QTFF Metadata) — the value inside one item.
const DATA: FourCc = boxes::fourcc(b"data");
/// `name` freeform key atom (QTFF Metadata) — the `----` item's key.
const NAME: FourCc = boxes::fourcc(b"name");

/// The `©` (0xA9) octet that prefixes Apple's original QuickTime user-data keys (`©nam`,
/// `©ART`, …). It is a single Mac OS Roman byte, *not* the two-octet UTF-8 encoding of U+00A9,
/// so the four-CC stays four octets (QTFF Metadata).
const COPYRIGHT: u8 = 0xA9;

// Well-known `data` type indicators (QTFF Metadata, "Well-Known Types"). The scanner needs
// only these five; every other type is skipped.
/// 0 — implicit / binary: the item's own definition says how to read the payload.
const TYPE_IMPLICIT: u32 = 0;
/// 1 — UTF-8 text (no NUL terminator; the box length is the length).
const TYPE_UTF8: u32 = 1;
/// 13 — a JPEG image.
const TYPE_JPEG: u32 = 13;
/// 14 — a PNG image.
const TYPE_PNG: u32 = 14;
/// 21 — a big-endian signed integer, 1/2/4/8 octets wide (the payload length is the width).
const TYPE_BE_SIGNED_INT: u32 = 21;

/// Longest `----` freeform key we will uppercase into the caller's scratch arena. Real keys
/// are short (`replaygain_track_gain` is 21 octets); the cap keeps a crafted file from turning
/// a single `name` box into a multi-megabyte arena allocation. A longer key is skipped.
const MAX_FREEFORM_KEY: usize = 128;

// =====================================================================================
// 1. Locating `moov` from a bounded prefix read
// =====================================================================================

/// Where the `moov` box is, relative to a prefix the caller has already read.
///
/// A scanner reads one slot-sized chunk of the file head and asks this question. After a
/// `faststart` remux `moov` is right after `ftyp` and the answer is [`InPrefix`](Self::InPrefix)
/// — the tags can be parsed from bytes already in hand. Straight out of most encoders `moov`
/// *trails* the `mdat`, so the answer is [`Beyond`](Self::Beyond) and the caller does one more
/// read of exactly the reported range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoovExtent {
    /// The whole `moov` box lies inside the prefix: `prefix[offset..offset + len]` is exactly
    /// the slice [`parse_moov_tags`] / [`props_from_moov`] want (header included).
    InPrefix { offset: usize, len: usize },
    /// **Read `file[offset..offset + len]` and look again** — the range starts on a top-level
    /// box boundary and lies (at least partly) outside the prefix. Two shapes reach here:
    ///
    /// - the `moov` **header was read** and the box merely runs past the end of the prefix —
    ///   then the range *is* the `moov` box, ready for [`parse_moov_tags`];
    /// - the walk stepped over a box (typically a multi-gigabyte `mdat`) and landed past the
    ///   end of the prefix — then the range is the **unread tail**, which is where `moov`
    ///   will be if the file has one. This is the ordinary non-`faststart` layout, so it must
    ///   not be a dead end.
    ///
    /// Feeding the re-read bytes back to `locate_moov` handles both uniformly (an exact
    /// `moov` box answers `InPrefix { offset: 0, .. }`), and the range always contains bytes
    /// the caller does not already have, so the loop cannot spin.
    Beyond { offset: u64, len: u64 },
    /// No `moov` and nothing left to read: the file is not ISO-BMFF, a box declared a
    /// structurally impossible length, or the walk reached the end of the file.
    NotFound,
}

/// Walk the top-level boxes of `prefix` (§4.2) and report where `moov` is. `file_len` is the
/// total file size, needed both to resolve a `size == 0` box ("to the end of the file", §4.2)
/// and to reject a box whose declared length runs past the file.
///
/// Never panics and never reads outside `prefix`; a malformed length simply ends the walk with
/// [`MoovExtent::NotFound`].
pub fn locate_moov(prefix: &[u8], file_len: u64) -> MoovExtent {
    let mut at = 0u64;
    while at < file_len {
        let Ok(pos) = usize::try_from(at) else { return MoovExtent::NotFound };
        match probe_box(prefix, pos, file_len) {
            Probe::Found(kind, total) => {
                // `total >= 8` (checked in `probe_box`), so `at` strictly increases — an
                // adversarial file cannot spin this loop.
                let end = at.saturating_add(total);
                if end > file_len {
                    return MoovExtent::NotFound; // runs past the file: give up, don't guess
                }
                if kind == boxtype::MOOV {
                    return if end <= prefix.len() as u64 {
                        MoovExtent::InPrefix { offset: pos, len: total as usize }
                    } else {
                        MoovExtent::Beyond { offset: at, len: total }
                    };
                }
                at = end;
            }
            // The header is out of reach. Everything from here to EOF is unread, and that is
            // where a trailing `moov` lives — unless the caller already holds the whole file,
            // in which case another read would return the same bytes (and spin the caller).
            Probe::Unread => {
                return if file_len > prefix.len() as u64 {
                    MoovExtent::Beyond { offset: at, len: file_len - at }
                } else {
                    MoovExtent::NotFound
                }
            }
            Probe::Malformed => return MoovExtent::NotFound,
        }
    }
    MoovExtent::NotFound
}

/// The outcome of probing one top-level box header during the prefix walk.
enum Probe {
    /// The header was read: the box type and its **total** length including the header (§4.2).
    Found(FourCc, u64),
    /// The header is (partly) outside the prefix — nothing can be said without another read.
    Unread,
    /// Structurally impossible: a box shorter than its own header, or a `size == 0` box whose
    /// offset is past the declared end of the file.
    Malformed,
}

/// Probe the box header at `prefix[at]` (§4.2) **without requiring its body to be present**.
///
/// [`boxes::read_box_header`] deliberately refuses this case: it hands out in-bounds body
/// windows, so a box with `end > data.len()` is an error — and a prefix walk hits exactly that
/// on the first `mdat` of a non-`faststart` file. It also resolves `size == 0` against the
/// buffer length, which for a prefix is the wrong answer (§4.2 says "to the end of the
/// *file*"). So this reads the same three header fields — `size`, `type`, and the 64-bit
/// `largesize` that follows when `size == 1` — and yields the declared length instead of a
/// body. It is not a second box walker: nothing below the top level is read here, and once the
/// caller has the complete `moov` the shared [`boxes::for_each_child`] / [`boxes::find_child`]
/// do all of the tree work.
fn probe_box(prefix: &[u8], at: usize, file_len: u64) -> Probe {
    let Some(head) = prefix.get(at..at.saturating_add(8)) else { return Probe::Unread };
    let size32 = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as u64;
    let kind: FourCc = [head[4], head[5], head[6], head[7]];
    let (header_len, total) = match size32 {
        // size == 1: a 64-bit `largesize` follows the type (§4.2).
        1 => {
            let Some(ls) = prefix.get(at + 8..at.saturating_add(16)) else { return Probe::Unread };
            (16u64, u64::from_be_bytes([ls[0], ls[1], ls[2], ls[3], ls[4], ls[5], ls[6], ls[7]]))
        }
        // size == 0: the box extends to the end of the file (§4.2) — hence `file_len`, which
        // is the whole reason this cannot go through `read_box_header`.
        0 => match file_len.checked_sub(at as u64) {
            Some(n) => (8, n),
            None => return Probe::Malformed,
        },
        n => (8, n),
    };
    // A `uuid` box's 16-octet usertype sits inside the payload; the *total* length already
    // covers it, and the top-level walk only needs that length, so it needs no special case.
    if total < header_len {
        return Probe::Malformed;
    }
    Probe::Found(kind, total)
}

// =====================================================================================
// 2. iTunes metadata: moov → udta → meta → ilst
// =====================================================================================

/// Parse the iTunes metadata in a `moov` box, pushing canonical tags into `sink`.
///
/// **Convention: `moov` is the *complete* box, header included** — byte-for-byte the slice
/// [`locate_moov`] describes ([`MoovExtent::InPrefix`] indexes straight into the prefix;
/// [`MoovExtent::Beyond`] names the range to re-read). A slice whose first box is not `moov`
/// is ignored, so a caller cannot accidentally pass a payload and get garbage.
///
/// The path is `moov` → `udta` (§8.10.1) → `meta` (§8.11.1) → `ilst` (QTFF Metadata), with a
/// fallback to a `meta` sitting directly under `moov` (both placements are legal — §8.11.1
/// allows `meta` at file, movie and track level — and a few writers use the latter).
///
/// `scratch` backs the one variable-length key this parser materialises (a `----` freeform
/// name, uppercased); text values are emitted as borrows of `moov` itself, and picture bytes
/// are the input slice verbatim. Per the [`TagSink`] contract every argument is borrowed for
/// the call only — a sink that keeps anything copies it.
pub fn parse_moov_tags(moov: &[u8], scratch: &Arena, sink: &mut impl TagSink) {
    let Some(body) = moov_body(moov) else { return };

    // `moov` → `udta` → `meta` (where iTunes writes it), else `moov` → `meta`.
    let via_udta = boxes::find_child(body, UDTA).ok().flatten().and_then(|u| {
        let udta = u.body(body);
        boxes::find_child(udta, META).ok().flatten().map(|m| m.body(udta))
    });
    let meta = via_udta
        .or_else(|| boxes::find_child(body, META).ok().flatten().map(|m| m.body(body)));
    let Some(meta) = meta else { return };
    let Some(ilst) = ilst_children(meta) else { return };

    // Each child of `ilst` is one item; its *box type* is the item id (QTFF Metadata). A
    // malformed item ends the walk (`for_each_child` propagates), keeping whatever preceded it.
    let _ = boxes::for_each_child(ilst, |item| {
        parse_item(item.kind, item.body(ilst), scratch, sink);
        Ok(())
    });
}

/// The payload of a complete `moov` box (see [`parse_moov_tags`]'s convention). `None` if the
/// slice does not start with a well-formed `moov` header.
fn moov_body(moov: &[u8]) -> Option<&[u8]> {
    let h = boxes::read_box_header(moov, 0).ok()?;
    (h.kind == boxtype::MOOV).then(|| h.body(moov))
}

/// The `ilst` payload inside a `meta` box payload.
///
/// **`meta` is a FullBox (§8.11.1): four octets of `version` + `flags` precede its children.**
/// Walking a `meta` payload from offset 0 reads every child header four octets early and gets
/// nonsense sizes and types — the classic MP4 metadata bug, and the reason this is its own
/// documented function with its own test.
///
/// QuickTime-heritage files (`.mov`, plus a few muxers) write `meta` as a plain container with
/// no version/flags. We probe the spec layout first and fall back to offset 0 only when it
/// yields no `ilst`; both probes go through the bounds-checked [`boxes::find_child`], so a
/// wrong guess finds nothing rather than misreading anything.
fn ilst_children(meta: &[u8]) -> Option<&[u8]> {
    for children in [meta.get(4..)?, meta] {
        if let Ok(Some(h)) = boxes::find_child(children, ILST) {
            return Some(h.body(children));
        }
    }
    None
}

/// Dispatch one `ilst` item (`kind` = the item's box type, `item` = its payload).
fn parse_item(kind: FourCc, item: &[u8], scratch: &Arena, sink: &mut impl TagSink) {
    match &kind {
        b"trkn" => number_pair(item, "TRACKNUMBER", sink),
        b"disk" => number_pair(item, "DISCNUMBER", sink),
        b"covr" => cover(item, sink),
        b"----" => freeform(item, scratch, sink),
        // `gnre` is the *legacy* genre item: an implicit u16 holding an ID3v1 genre index
        // plus one (QTFF Metadata). Resolving it would mean carrying ID3v1's 80-odd
        // informally-extended genre names, and a file that has `gnre` almost always also has
        // `©gen` with the same genre as text — which this parser emits. So it is skipped by
        // design rather than by omission.
        b"gnre" => {}
        _ => {
            if let Some(key) = text_key(kind) {
                text_item(item, key, sink);
            }
        }
    }
}

/// The canonical Vorbis-comment key for an Apple text item id (QTFF Metadata), or `None` for
/// an item this scanner does not carry (`tmpo`, `cpil`, `stik`, `purl`, …). Unknown ids are
/// skipped silently — an MP4 written by a video tool is full of them.
fn text_key(kind: FourCc) -> Option<&'static str> {
    Some(match kind {
        [COPYRIGHT, b'n', b'a', b'm'] => "TITLE",
        [COPYRIGHT, b'A', b'R', b'T'] => "ARTIST",
        [COPYRIGHT, b'a', b'l', b'b'] => "ALBUM",
        [b'a', b'A', b'R', b'T'] => "ALBUMARTIST",
        [COPYRIGHT, b'd', b'a', b'y'] => "DATE",
        [COPYRIGHT, b'g', b'e', b'n'] => "GENRE",
        [COPYRIGHT, b'w', b'r', b't'] => "COMPOSER",
        [COPYRIGHT, b'c', b'm', b't'] => "COMMENT",
        _ => return None,
    })
}

/// Call `f(well_known_type, value)` for every `data` box inside an `ilst` item.
///
/// A *metadata item data atom* (QTFF Metadata) begins with a 4-octet **type indicator** — a
/// one-octet type-set indicator (0 = the well-known set) then a three-octet type number — and a
/// 4-octet **locale indicator** (country/language; a scanner has one locale to show, so it is
/// read past, not used). The value is everything after those eight octets. An item may hold
/// several `data` boxes; each is delivered, which is how a repeated key reaches the sink.
fn for_each_data<F: FnMut(u32, &[u8])>(item: &[u8], mut f: F) {
    let _ = boxes::for_each_child(item, |h| {
        if h.kind == DATA {
            let d = h.body(item);
            if let Some(head) = d.get(..8) {
                let indicator = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
                // Only the well-known type set (indicator octet 0) is defined for us; a
                // vendor-specific set would need that vendor's registry.
                if indicator >> 24 == 0 {
                    f(indicator & 0x00FF_FFFF, &d[8..]);
                }
            }
        }
        Ok(())
    });
}

/// A mapped text item: emit its UTF-8 payload verbatim (a borrow of the input — MP4 text is
/// already UTF-8, so no transcoding scratch is needed), or, for the handful of taggers that
/// store a mapped key as a well-known integer (a year, a rating), its decimal rendering.
/// Invalid UTF-8 is dropped rather than replaced — a scanner must not invent characters.
fn text_item(item: &[u8], key: &'static str, sink: &mut impl TagSink) {
    for_each_data(item, |ty, payload| match ty {
        TYPE_UTF8 => {
            if let Ok(s) = core::str::from_utf8(payload) {
                sink.text(key, s);
            }
        }
        TYPE_BE_SIGNED_INT => {
            if let Some(n) = be_signed_int(payload) {
                let mut s = FixedStr::<24>::new();
                let _ = write!(s, "{n}");
                if let Some(v) = s.as_str() {
                    sink.text(key, v);
                }
            }
        }
        _ => {}
    });
}

/// `trkn` / `disk` (QTFF Metadata): an implicit-typed binary payload of 2 reserved octets, the
/// 1-based `index` (u16 BE), the `total` (u16 BE), then padding. Emitted the way Vorbis
/// comments carry it — `"3/12"`, or plain `"3"` when the total is absent (0) — so a consumer
/// sees one convention across FLAC, Ogg, Matroska and MP4. A zero index means "not set" and is
/// dropped.
fn number_pair(item: &[u8], key: &'static str, sink: &mut impl TagSink) {
    for_each_data(item, |ty, payload| {
        if ty != TYPE_IMPLICIT {
            return;
        }
        let Some(b) = payload.get(..6) else { return };
        let index = u16::from_be_bytes([b[2], b[3]]);
        let total = u16::from_be_bytes([b[4], b[5]]);
        if index == 0 {
            return;
        }
        // Stack-formatted: "65535/65535" is 11 octets, so 16 cannot overflow.
        let mut s = FixedStr::<16>::new();
        let _ = if total == 0 { write!(s, "{index}") } else { write!(s, "{index}/{total}") };
        if let Some(v) = s.as_str() {
            sink.text(key, v);
        }
    });
}

/// `covr` cover art (QTFF Metadata): one picture per `data` box, typed 13 (JPEG) or 14 (PNG).
/// Writers that use the implicit type (0) are common, so the payload magic is sniffed instead:
/// JPEG's `FF D8` SOI marker (ITU-T T.81 §B.1.1.3) or PNG's 8-octet signature, whose first four
/// octets `89 50 4E 47` are enough (ISO/IEC 15948 §5.2). An unrecognised type or magic is
/// skipped — the sink's `mime` must never be a guess.
fn cover(item: &[u8], sink: &mut impl TagSink) {
    for_each_data(item, |ty, payload| {
        let mime = match ty {
            TYPE_JPEG => "image/jpeg",
            TYPE_PNG => "image/png",
            TYPE_IMPLICIT => match payload {
                [0xFF, 0xD8, ..] => "image/jpeg",
                [0x89, 0x50, 0x4E, 0x47, ..] => "image/png",
                _ => return,
            },
            _ => return,
        };
        if !payload.is_empty() {
            sink.picture(mime, payload);
        }
    });
}

/// A `----` freeform item (QTFF Metadata): three children — `mean` (the reverse-DNS namespace,
/// e.g. `com.apple.iTunes`), `name` (the key, e.g. `replaygain_track_gain`) and `data` (the
/// value). `mean` and `name` are *full* atoms: four octets of version/flags precede the UTF-8
/// string, which is not NUL-terminated.
///
/// The emitted key is the ASCII-uppercased `name` — which is exactly how `REPLAYGAIN_TRACK_GAIN`
/// and friends arrive in the canonical vocabulary, since ReplayGain has no Apple item id and
/// every tagger writes it as a freeform pair. The namespace is deliberately *not* matched
/// against `com.apple.iTunes`: other taggers write the same keys under their own reverse-DNS
/// namespace, and dropping those would lose the tag the caller actually asked for.
///
/// Uppercasing needs a buffer of the key's length, so this is the one place a scratch [`Arena`]
/// is used. Only ASCII letters change case, so the copy stays valid UTF-8 by construction.
fn freeform(item: &[u8], scratch: &Arena, sink: &mut impl TagSink) {
    let Some(h) = boxes::find_child(item, NAME).ok().flatten() else { return };
    // Past the full-atom version/flags: the raw key octets.
    let Some(name) = h.body(item).get(4..) else { return };
    if name.is_empty() || name.len() > MAX_FREEFORM_KEY {
        return;
    }
    let buf = scratch.alloc_bytes(name.len());
    buf.copy_from_slice(name);
    buf.make_ascii_uppercase();
    let Ok(key) = core::str::from_utf8(buf) else { return };
    for_each_data(item, |ty, payload| {
        if ty == TYPE_UTF8 {
            if let Ok(v) = core::str::from_utf8(payload) {
                sink.text(key, v);
            }
        }
    });
}

/// A well-known type-21 payload as an `i64`. Apple writes these 1, 2, 4 or 8 octets wide and
/// the payload length *is* the width (QTFF Metadata); any other length is not an integer we can
/// read.
fn be_signed_int(payload: &[u8]) -> Option<i64> {
    Some(match payload.len() {
        1 => payload[0] as i8 as i64,
        2 => i16::from_be_bytes([payload[0], payload[1]]) as i64,
        4 => i32::from_be_bytes(payload.try_into().ok()?) as i64,
        8 => i64::from_be_bytes(payload.try_into().ok()?),
        _ => return None,
    })
}

/// A fixed-capacity formatting buffer — the alloc-free stand-in for `format!` on the parse path
/// (`write!` needs a [`core::fmt::Write`] sink, and the crate's allocation discipline keeps
/// `String` out of here). Overflow is **sticky**: the buffer marks itself invalid and
/// [`as_str`](Self::as_str) returns `None`, so a half-written number is never emitted as a tag.
struct FixedStr<const N: usize> {
    buf: [u8; N],
    len: usize,
    ok: bool,
}

impl<const N: usize> FixedStr<N> {
    fn new() -> Self {
        Self { buf: [0; N], len: 0, ok: true }
    }

    /// The formatted text, or `None` if a write overflowed the buffer.
    fn as_str(&self) -> Option<&str> {
        self.ok.then(|| core::str::from_utf8(&self.buf[..self.len]).ok())?
    }
}

impl<const N: usize> core::fmt::Write for FixedStr<N> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        match self.buf.get_mut(self.len..self.len + s.len()) {
            Some(dst) => {
                dst.copy_from_slice(s.as_bytes());
                self.len += s.len();
                Ok(())
            }
            None => {
                self.ok = false;
                Err(core::fmt::Error)
            }
        }
    }
}

// =====================================================================================
// 3. No-decode props: duration, sample rate, channels
// =====================================================================================

/// What a listing view shows without decoding a single sample. Every field is optional: a
/// fragmented file (`mvex`, `mvhd.duration == 0`) has no duration in its `moov`, and a
/// video-only file has no rate or channels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mp4Props {
    /// Presentation duration in nanoseconds.
    pub duration_ns: Option<u64>,
    /// Audio sample rate in Hz, from the sound track's sample entry.
    pub sample_rate: Option<u32>,
    /// Audio channel count, from the sound track's sample entry.
    pub channels: Option<u32>,
}

/// Best-effort duration/rate/channels from a complete `moov` box (same convention as
/// [`parse_moov_tags`]: header included).
///
/// Rate and channels come off the first sample entry of the sound track's `stsd`
/// (§8.5.2 → §12.2.3). Duration is the harder question, and an MP4 answers it three times over.
///
/// ## The three statements of length, and which one is the file's playing time
///
/// 1. The sound track's `mdhd` (§8.4.2) — its **media** duration, counted in the fine media
///    timescale (44 100, 48 000). Every sample the track stores, *including* the codec priming
///    an edit list exists to throw away.
/// 2. Its `edts`/`elst` (§8.6.6) — its **presentation** duration: the sum of the
///    `segment_duration`s of the edits that actually present media, counted in the coarse
///    *movie* timescale. This is what a player's timeline shows, and what `ffprobe` reports as
///    that stream's duration.
/// 3. The `mvhd` (§8.2.2) — the presentation duration of the **longest track in the movie**,
///    also in the movie timescale. This is what `ffprobe` reports as the *format* duration.
///
/// They disagree, and each disagreement means something different, so the choice is made twice:
///
/// **Media against presentation.** An edit list that trims priming makes (2) genuinely shorter
/// than (1) — the samples exist but are not presented. But the overwhelmingly common `elst` is
/// a single `media_time == 0` entry that trims nothing and merely restates (1) rounded to the
/// movie timescale, and *there* (1) is the better number because it is the finer one. So the
/// edit list wins only when the two differ by **more than one movie tick** — a gap smaller than
/// the movie timescale's own resolution is a rounding artefact, not a trim. Measured over the
/// 15 MP4/M4A files in the reference library, this reproduces `ffprobe`'s per-stream duration
/// exactly on all 15; taking the edit list unconditionally would put four of them 0.17–0.87 ms
/// *off*, because their `elst` is that rounded restatement.
///
/// **Track against movie.** A sound track is not necessarily the longest track. When the movie
/// holds **more than one track** and `mvhd` outlasts the sound track by more than one movie
/// tick, some other track — video, in practice — is what the file's playing time actually is,
/// and (3) is reported. The track-count gate is what makes that inference safe: §8.2.2 defines
/// `mvhd.duration` as the longest track's, so in a single-track file it *is* the sound track's,
/// and any excess is a stale or wrong movie header rather than evidence of anything. The
/// one-tick guard then keeps the coarse movie header from displacing a finer, equally true
/// track duration.
///
/// Both rules are no-ops on a single-track audio file with no edit list, which is the shape of
/// almost every file in a music library: it reports its `mdhd`, exactly as before.
///
/// Deliberately **not** built on [`Mp4Reader`](crate::Mp4Reader): that rejects fragmented files
/// outright, while a scanner should still report what a fragmented file's `moov` does say (and
/// leave `duration_ns: None` when it says nothing). This is a manual, allocation-free walk that
/// never errors.
pub fn props_from_moov(moov: &[u8]) -> Mp4Props {
    let mut props = Mp4Props::default();
    let Some(body) = moov_body(moov) else { return props };

    // The movie header comes first: its timescale is the unit *every* `elst` segment duration
    // is counted in (§8.6.6), so no edit list below can be read without it.
    let mvhd = boxes::find_child(body, boxtype::MVHD)
        .ok()
        .flatten()
        .and_then(|h| boxes::parse_mvhd(h.body(body)).ok());
    let movie_ns = mvhd.and_then(|m| ticks_to_ns(m.duration, m.timescale));
    let tick_ns = mvhd.and_then(|m| movie_tick_ns(m.timescale));

    // The first track whose `mdia` declares handler `soun` (§8.4.3). The enclosing `trak`
    // travels with it: the edit list that turns this track's media timeline into a
    // presentation timeline is a sibling of `mdia`, not a child of it (§8.6.5).
    let mut soun: Option<(&[u8], &[u8])> = None;
    let mut tracks = 0usize;
    let _ = boxes::for_each_child(body, |h| {
        if h.kind == boxtype::TRAK {
            tracks += 1;
            let trak = h.body(body);
            if soun.is_none() {
                if let Ok(Some(m)) = boxes::find_child(trak, boxtype::MDIA) {
                    let mdia = m.body(trak);
                    if handler_type(mdia) == Some(*b"soun") {
                        soun = Some((trak, mdia));
                    }
                }
            }
        }
        Ok(())
    });

    if let Some((trak, mdia)) = soun {
        let media_ns = boxes::find_child(mdia, boxtype::MDHD)
            .ok()
            .flatten()
            // `parse_mdhd` handles version 0 (32-bit times) vs version 1 (64-bit) per §8.4.2.
            .and_then(|h| boxes::parse_mdhd(h.body(mdia)).ok())
            .and_then(|m| ticks_to_ns(m.duration, m.timescale));
        let edit_ns = mvhd.and_then(|m| edit_duration_ns(trak, m.timescale));
        props.duration_ns = pick_track_duration(media_ns, edit_ns, tick_ns);

        if let Some((rate, channels)) = audio_format(mdia) {
            props.sample_rate = (rate != 0).then_some(rate);
            props.channels = (channels != 0).then_some(channels);
        }
    }

    match props.duration_ns {
        // §8.2.2: `mvhd.duration` is the *longest* track's presentation duration. In a
        // multi-track movie, longer than the sound track by more than the movie timescale can
        // resolve means another track outlasts the audio, and that — not the audio — is how
        // long the file plays. In a single-track one the two describe the same track, so a
        // disagreement is a wrong movie header and the track's own word stands.
        Some(track) if tracks > 1 => {
            if let (Some(movie), Some(tick)) = (movie_ns, tick_ns) {
                if movie > track.saturating_add(tick) {
                    props.duration_ns = Some(movie);
                }
            }
        }
        Some(_) => {}
        // No sound track, or one whose media header says nothing.
        None => props.duration_ns = movie_ns,
    }
    props
}

/// One movie-timescale tick in nanoseconds, rounded up — the resolution at which `mvhd` and
/// every `elst` state a duration (§8.2.2, §8.6.6), and therefore the size of a difference
/// between one of them and a finer media-timescale duration that means nothing.
///
/// Rounded **up** so the guard is inclusive: a movie timescale of 600 ticks/s resolves to
/// 1 666 667 ns, and a restated duration may land a whole tick away in either direction.
fn movie_tick_ns(movie_timescale: u32) -> Option<u64> {
    (movie_timescale != 0).then(|| 1_000_000_000u64.div_ceil(movie_timescale as u64))
}

/// Reconcile a sound track's media duration with its edit list's presentation duration.
///
/// The edit list is authoritative when it says something the media header does not — a trim of
/// more than one movie tick. Otherwise the media header is preferred as the finer-grained
/// statement of the same length. See [`props_from_moov`] for why, and for the measurement.
fn pick_track_duration(media_ns: Option<u64>, edit_ns: Option<u64>, tick_ns: Option<u64>) -> Option<u64> {
    match (media_ns, edit_ns) {
        (Some(media), Some(edit)) => {
            let tick = tick_ns.unwrap_or(0);
            Some(if media.abs_diff(edit) > tick { edit } else { media })
        }
        (media, edit) => media.or(edit),
    }
}

/// A track's **presentation** duration from its edit list (§8.6.5 `edts` → §8.6.6 `elst`), in
/// nanoseconds. `None` when the track has no edit list, or none that presents any media.
fn edit_duration_ns(trak: &[u8], movie_timescale: u32) -> Option<u64> {
    let edts = boxes::find_child(trak, boxtype::EDTS).ok()??;
    let edts = edts.body(trak);
    let elst = boxes::find_child(edts, boxtype::ELST).ok()??;
    ticks_to_ns(elst_presented_ticks(elst.body(edts))?, movie_timescale)
}

/// Sum the `segment_duration`s of an `elst` payload's edits that present media (§8.6.6), in
/// movie-timescale ticks. `None` for a list this parser cannot trust: an unknown version, a
/// declared `entry_count` the box is too short to carry, or a list of nothing but empty edits.
///
/// An **empty edit** (`media_time == -1`) presents no media — §8.6.6 defines it as a gap, used
/// to delay a track's start. It contributes to where later edits land on the presentation
/// timeline but not to how much media is presented, so it is skipped here. That is the same
/// convention [`crate::reader`]'s `edit_list_shift` follows, and it matches what `ffprobe`
/// reports for a stream whose edit list opens with one.
///
/// A deliberately hand-rolled walk rather than [`boxes::parse_elst`]: that one returns a
/// `Vec`, and this is the scanner path, which allocates nothing (spec: allocation discipline).
fn elst_presented_ticks(body: &[u8]) -> Option<u64> {
    let version = *body.first()?;
    let count = u32::from_be_bytes(body.get(4..8)?.try_into().ok()?) as usize;
    // Entry widths per §8.6.6: v0 is u32 + i32 + the 16.16 `media_rate` pair; v1 widens the
    // first two to 64 bits. A version this parser does not know describes entries of an
    // unknown width, so there is no safe way to walk past even the first one.
    let width = match version {
        0 => 12usize,
        1 => 20usize,
        _ => return None,
    };
    let entries = body.get(8..)?;
    // Bound the walk by the bytes actually present before looping: `entry_count` is attacker
    // -controlled and a `u32::MAX` claim must cost a compare, not four billion iterations.
    if count > entries.len() / width {
        return None;
    }

    let mut total = 0u64;
    let mut presented = 0usize;
    for i in 0..count {
        let e = entries.get(i * width..(i + 1) * width)?;
        let (segment_duration, media_time) = if version == 1 {
            (
                u64::from_be_bytes(e.get(0..8)?.try_into().ok()?),
                i64::from_be_bytes(e.get(8..16)?.try_into().ok()?),
            )
        } else {
            (
                u32::from_be_bytes(e.get(0..4)?.try_into().ok()?) as u64,
                i32::from_be_bytes(e.get(4..8)?.try_into().ok()?) as i64,
            )
        };
        if media_time != -1 {
            total = total.checked_add(segment_duration)?;
            presented += 1;
        }
    }
    (presented != 0).then_some(total)
}

/// The `handler_type` declared by a `mdia`'s `hdlr` (§8.4.3): FullBox version/flags (4 octets),
/// a `pre_defined` u32 (4), then the four-CC — `soun` for audio, `vide` for video.
fn handler_type(mdia: &[u8]) -> Option<FourCc> {
    let h = boxes::find_child(mdia, HDLR).ok()??;
    let b = h.body(mdia).get(8..12)?;
    Some([b[0], b[1], b[2], b[3]])
}

/// `(sample_rate, channels)` from the first sample entry of a sound track's `stsd`
/// (`mdia` → `minf` → `stbl` → `stsd`, §8.4.4/§8.5.1/§8.5.2).
///
/// An `AudioSampleEntry` (§12.2.3) begins with the SampleEntry base (6 reserved octets + a
/// `data_reference_index`, §8.5.2.2), 8 reserved octets, then `channelcount` (u16) at octet 16,
/// `samplesize` (u16), 4 octets pre_defined/reserved, and `samplerate` at octet 24 as a 16.16
/// fixed-point value whose integer part is the rate in Hz. That prologue is identical for
/// every audio four-CC — `mp4a`, `alac`, `Opus`, `fLaC` — so it is read straight off the entry
/// without dispatching on the codec; the codec-specific children (`esds`, `dOps`, `alac`) are
/// [`crate::codec`]'s business, not a scanner's.
///
/// One deviation, from QuickTime heritage: a *version 2* sound description (the version lives
/// at octet 8, inside what §12.2.3 calls reserved) parks a placeholder in the 16.16 field and
/// carries the true rate as an IEEE-754 `f64` at octet 32 with the channel count as a u32 at
/// octet 40. That layout is read when the version says so, since it is the only way a
/// high-rate or many-channel `.mov` reports its real format.
fn audio_format(mdia: &[u8]) -> Option<(u32, u32)> {
    let minf = boxes::find_child(mdia, boxtype::MINF).ok()??;
    let minf = minf.body(mdia);
    let stbl = boxes::find_child(minf, boxtype::STBL).ok()??;
    let stbl = stbl.body(minf);
    let stsd = boxes::find_child(stbl, boxtype::STSD).ok()??;
    // `stsd` is a FullBox: version/flags (4) + entry_count (4) precede the entries (§8.5.2).
    let entries = stsd.body(stbl).get(8..)?;
    let entry = boxes::read_box_header(entries, 0).ok()?;
    let e = entry.body(entries);

    let version = u16::from_be_bytes([*e.get(8)?, *e.get(9)?]);
    if version == 2 {
        let rate = f64::from_bits(u64::from_be_bytes(e.get(32..40)?.try_into().ok()?));
        let channels = u32::from_be_bytes(e.get(40..44)?.try_into().ok()?);
        // A non-finite or negative rate is a corrupt field, not a format.
        let rate = if rate.is_finite() && rate > 0.0 { rate as u32 } else { 0 };
        return Some((rate, channels));
    }
    let channels = u16::from_be_bytes([*e.get(16)?, *e.get(17)?]) as u32;
    let rate = u32::from_be_bytes(e.get(24..28)?.try_into().ok()?) >> 16;
    Some((rate, channels))
}

/// Media ticks → nanoseconds (`duration * 1e9 / timescale`), computed in `u128` so a 64-bit
/// duration at a fine timescale cannot overflow.
///
/// `None` when the duration is not declared: a zero timescale or duration, or the all-ones
/// sentinel §8.2.2/§8.4.2 define as "unknown" — which is exactly what a fragmented file's
/// `moov` (or a still-being-written one) reports.
fn ticks_to_ns(duration: u64, timescale: u32) -> Option<u64> {
    if timescale == 0 || duration == 0 || duration == u32::MAX as u64 || duration == u64::MAX {
        return None;
    }
    u64::try_from(duration as u128 * 1_000_000_000 / timescale as u128).ok()
}

#[cfg(test)]
// Fixture assembly and the collecting sink are test-only heap use (spec: allocation discipline
// — tests are a sanctioned exception); the parsers under test allocate nothing.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------------
    // Fixture writers (the `boxes::tests::boxed` pattern, extended for full boxes and the
    // Apple item scheme).
    // ---------------------------------------------------------------------------------

    /// A box: 8-octet header (size, type) then `body`.
    fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = ((8 + body.len()) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    /// A FullBox: version 0, flags 0, then `body` (§4.2).
    fn full_bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut b = vec![0u8; 4];
        b.extend_from_slice(body);
        bx(kind, &b)
    }

    /// A metadata item `data` atom: type indicator (well-known set) + locale + value.
    fn data_box(ty: u32, value: &[u8]) -> Vec<u8> {
        let mut b = ty.to_be_bytes().to_vec();
        b.extend_from_slice(&0u32.to_be_bytes()); // locale indicator
        b.extend_from_slice(value);
        bx(b"data", &b)
    }

    /// An `ilst` item holding one `data` atom.
    fn item(kind: &[u8; 4], ty: u32, value: &[u8]) -> Vec<u8> {
        bx(kind, &data_box(ty, value))
    }

    /// `moov` → `udta` → `meta`(FullBox, with the `hdlr` real files carry) → `ilst`(items).
    fn moov_with_items(items: &[Vec<u8>]) -> Vec<u8> {
        let mut ilst = Vec::new();
        for i in items {
            ilst.extend_from_slice(i);
        }
        // The metadata handler declaration iTunes writes first (§8.4.3): pre_defined, then
        // handler_type `mdir`, then the reserved trio and an empty name.
        let mut hdlr_body = vec![0u8; 4];
        hdlr_body.extend_from_slice(b"mdir");
        hdlr_body.extend_from_slice(b"appl");
        hdlr_body.extend_from_slice(&[0u8; 9]);
        let mut meta_body = bx(b"hdlr", &hdlr_body);
        meta_body.extend_from_slice(&bx(b"ilst", &ilst));
        let meta = full_bx(b"meta", &meta_body);
        bx(b"moov", &bx(b"udta", &meta))
    }

    /// A collecting [`TagSink`] — the test-side twin of `TagListSink` without a pool.
    #[derive(Default)]
    struct Collect {
        text: Vec<(String, String)>,
        pics: Vec<(String, Vec<u8>)>,
    }

    impl TagSink for Collect {
        fn text(&mut self, key: &str, value: &str) {
            self.text.push((key.to_string(), value.to_string()));
        }

        fn picture(&mut self, mime: &str, data: &[u8]) {
            self.pics.push((mime.to_string(), data.to_vec()));
        }
    }

    impl Collect {
        fn get(&self, key: &str) -> Option<&str> {
            self.text.iter().find(|(k, _)| k == key).map(|(_, v)| &**v)
        }
    }

    fn tags_of(moov: &[u8]) -> Collect {
        let scratch = Arena::new(Arena::DEFAULT_CHUNK);
        let mut sink = Collect::default();
        parse_moov_tags(moov, &scratch, &mut sink);
        sink
    }

    // ---------------------------------------------------------------------------------
    // ilst
    // ---------------------------------------------------------------------------------

    /// `meta` is a FullBox (§8.11.1): its children start four octets in. The fixture is built
    /// so a naive non-FullBox walk finds nothing at all — that walk is run here as the
    /// control, then the real parser is asserted to read the tag.
    #[test]
    fn meta_fullbox_offset_is_honoured() {
        let moov = moov_with_items(&[item(b"\xA9nam", TYPE_UTF8, b"Enough")]);

        // The control: the `meta` payload walked from offset 0, as a parser that forgets the
        // FullBox prefix would. The first "size" it reads is the version/flags word (0), so
        // the walk swallows the whole payload as one box and never sees `ilst`.
        let udta = boxes::find_child(&moov[8..], UDTA).unwrap().unwrap();
        let udta = udta.body(&moov[8..]);
        let meta = boxes::find_child(udta, META).unwrap().unwrap();
        let meta = meta.body(udta);
        assert!(
            !matches!(boxes::find_child(meta, ILST), Ok(Some(_))),
            "a non-FullBox walk of `meta` must NOT find `ilst` — that is the bug under test"
        );

        // The parser reads the four octets and finds it.
        assert_eq!(tags_of(&moov).get("TITLE"), Some("Enough"));
    }

    /// A `meta` written the QuickTime way — no version/flags — still parses (the fallback).
    #[test]
    fn quicktime_non_fullbox_meta_still_parses() {
        let ilst = bx(b"ilst", &item(b"\xA9nam", TYPE_UTF8, b"Mov"));
        let moov = bx(b"moov", &bx(b"udta", &bx(b"meta", &ilst)));
        assert_eq!(tags_of(&moov).get("TITLE"), Some("Mov"));
    }

    /// Every mapped Apple text id lands on its canonical Vorbis-comment key, UTF-8 verbatim.
    #[test]
    fn text_items_map_to_canonical_keys() {
        let moov = moov_with_items(&[
            item(b"\xA9nam", TYPE_UTF8, "Enough".as_bytes()),
            item(b"\xA9ART", TYPE_UTF8, "Fred again..".as_bytes()),
            item(b"\xA9alb", TYPE_UTF8, "Actual Life 3".as_bytes()),
            item(b"aART", TYPE_UTF8, "Various Artists".as_bytes()),
            item(b"\xA9day", TYPE_UTF8, "2021-10-29".as_bytes()),
            item(b"\xA9gen", TYPE_UTF8, "Électronique".as_bytes()),
            item(b"\xA9wrt", TYPE_UTF8, "F. Gibson".as_bytes()),
            item(b"\xA9cmt", TYPE_UTF8, "ripped".as_bytes()),
            item(b"stik", TYPE_IMPLICIT, &[1]), // unmapped id: skipped silently
        ]);
        let t = tags_of(&moov);
        assert_eq!(t.get("TITLE"), Some("Enough"));
        assert_eq!(t.get("ARTIST"), Some("Fred again.."));
        assert_eq!(t.get("ALBUM"), Some("Actual Life 3"));
        assert_eq!(t.get("ALBUMARTIST"), Some("Various Artists"));
        assert_eq!(t.get("DATE"), Some("2021-10-29"));
        assert_eq!(t.get("GENRE"), Some("Électronique"), "non-ASCII UTF-8 passes through");
        assert_eq!(t.get("COMPOSER"), Some("F. Gibson"));
        assert_eq!(t.get("COMMENT"), Some("ripped"));
        assert_eq!(t.text.len(), 8, "the unmapped `stik` item is skipped");
        assert!(t.pics.is_empty());
    }

    /// A mapped id stored as a well-known integer (type 21) is rendered decimally.
    #[test]
    fn integer_typed_text_item_is_rendered_decimally() {
        let moov = moov_with_items(&[item(b"\xA9day", TYPE_BE_SIGNED_INT, &2015u16.to_be_bytes())]);
        assert_eq!(tags_of(&moov).get("DATE"), Some("2015"));
    }

    /// `trkn`/`disk`: index + total → `"N/M"`, and a zero total → `"N"`.
    #[test]
    fn trkn_and_disk_emit_index_and_total() {
        // reserved u16, index u16, total u16, reserved u16 (QTFF Metadata).
        let pair = |index: u16, total: u16| {
            let mut v = vec![0u8, 0];
            v.extend_from_slice(&index.to_be_bytes());
            v.extend_from_slice(&total.to_be_bytes());
            v.extend_from_slice(&[0, 0]);
            v
        };
        let moov = moov_with_items(&[
            item(b"trkn", TYPE_IMPLICIT, &pair(3, 12)),
            item(b"disk", TYPE_IMPLICIT, &pair(1, 2)),
        ]);
        let t = tags_of(&moov);
        assert_eq!(t.get("TRACKNUMBER"), Some("3/12"));
        assert_eq!(t.get("DISCNUMBER"), Some("1/2"));

        // total == 0 → the bare index, the Vorbis convention.
        let moov = moov_with_items(&[item(b"trkn", TYPE_IMPLICIT, &pair(3, 0))]);
        assert_eq!(tags_of(&moov).get("TRACKNUMBER"), Some("3"));

        // index == 0 → not set → nothing emitted.
        let moov = moov_with_items(&[item(b"trkn", TYPE_IMPLICIT, &pair(0, 12))]);
        assert!(tags_of(&moov).text.is_empty());
    }

    /// `covr`: a type-14 payload round-trips byte-exact as `image/png`, a type-13 one as
    /// `image/jpeg`, and an implicit-typed one is identified by its magic.
    #[test]
    fn covr_pictures_round_trip_byte_exact() {
        let png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\x00\x01";
        let jpeg = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00";
        let moov = moov_with_items(&[
            item(b"covr", TYPE_PNG, png),
            item(b"covr", TYPE_JPEG, jpeg),
            item(b"covr", TYPE_IMPLICIT, png),
            item(b"covr", TYPE_IMPLICIT, jpeg),
            item(b"covr", TYPE_IMPLICIT, b"not an image"),
            item(b"covr", 42, png), // unknown type: skipped
        ]);
        let t = tags_of(&moov);
        assert_eq!(
            t.pics,
            vec![
                ("image/png".to_string(), png.to_vec()),
                ("image/jpeg".to_string(), jpeg.to_vec()),
                ("image/png".to_string(), png.to_vec()),
                ("image/jpeg".to_string(), jpeg.to_vec()),
            ]
        );
        assert!(t.text.is_empty());
    }

    /// A `----` freeform item: `mean` + `name` + `data`, key uppercased — how ReplayGain
    /// arrives.
    #[test]
    fn freeform_replaygain_is_uppercased() {
        let mut ff = full_bx(b"mean", b"com.apple.iTunes");
        ff.extend_from_slice(&full_bx(b"name", b"replaygain_track_gain"));
        ff.extend_from_slice(&data_box(TYPE_UTF8, b"-7.35 dB"));
        let mut ff2 = full_bx(b"mean", b"org.example.tagger");
        ff2.extend_from_slice(&full_bx(b"name", b"replaygain_album_peak"));
        ff2.extend_from_slice(&data_box(TYPE_UTF8, b"0.988"));
        let moov = moov_with_items(&[bx(b"----", &ff), bx(b"----", &ff2)]);
        let t = tags_of(&moov);
        assert_eq!(t.get("REPLAYGAIN_TRACK_GAIN"), Some("-7.35 dB"));
        assert_eq!(
            t.get("REPLAYGAIN_ALBUM_PEAK"),
            Some("0.988"),
            "a non-Apple namespace still yields its key"
        );
    }

    /// `gnre` (the legacy ID3v1 genre index) is skipped; `©gen` carries the text.
    #[test]
    fn gnre_is_skipped_but_gen_is_kept() {
        let moov = moov_with_items(&[
            item(b"gnre", TYPE_IMPLICIT, &18u16.to_be_bytes()),
            item(b"\xA9gen", TYPE_UTF8, b"Techno"),
        ]);
        let t = tags_of(&moov);
        assert_eq!(t.text, vec![("GENRE".to_string(), "Techno".to_string())]);
    }

    /// A slice that is not a `moov` box, or has no metadata at all, emits nothing.
    #[test]
    fn missing_metadata_emits_nothing() {
        assert!(tags_of(&[]).text.is_empty());
        assert!(tags_of(&bx(b"ftyp", b"M4A ")).text.is_empty());
        assert!(tags_of(&bx(b"moov", &bx(b"udta", b""))).text.is_empty());
        assert!(tags_of(&bx(b"moov", &bx(b"udta", &full_bx(b"meta", b"")))).text.is_empty());
    }

    // ---------------------------------------------------------------------------------
    // locate_moov
    // ---------------------------------------------------------------------------------

    /// A faststart layout (`ftyp`, `moov`, `mdat`) read with the whole `moov` in the prefix.
    #[test]
    fn locate_moov_in_prefix() {
        let ftyp = bx(b"ftyp", b"M4A \0\0\0\0");
        let moov = moov_with_items(&[item(b"\xA9nam", TYPE_UTF8, b"Front")]);
        let mut file = ftyp.clone();
        file.extend_from_slice(&moov);
        file.extend_from_slice(&bx(b"mdat", &[0u8; 64]));
        let file_len = file.len() as u64;

        // The prefix stops inside the `mdat`, well past the end of `moov`.
        let prefix = &file[..ftyp.len() + moov.len() + 16];
        let MoovExtent::InPrefix { offset, len } = locate_moov(prefix, file_len) else {
            panic!("moov is entirely inside the prefix");
        };
        assert_eq!((offset, len), (ftyp.len(), moov.len()));
        // The reported window is exactly what the tag parser wants.
        assert_eq!(tags_of(&prefix[offset..offset + len]).get("TITLE"), Some("Front"));
    }

    /// The encoder-default layout: `moov` after a large `mdat`. With the `moov` header inside
    /// the prefix but its body cut off, the reported range is the box itself.
    #[test]
    fn locate_moov_after_mdat_reports_beyond() {
        let ftyp = bx(b"ftyp", b"isom\0\0\0\0");
        let mdat = bx(b"mdat", &vec![0u8; 4096]);
        let moov = moov_with_items(&[item(b"\xA9nam", TYPE_UTF8, b"Tail")]);
        let mut file = ftyp.clone();
        file.extend_from_slice(&mdat);
        file.extend_from_slice(&moov);
        let file_len = file.len() as u64;
        let moov_at = (ftyp.len() + mdat.len()) as u64;

        // A prefix that walks past `mdat` and reads the `moov` header, but not its body.
        let got = locate_moov(&file[..moov_at as usize + 12], file_len);
        assert_eq!(got, MoovExtent::Beyond { offset: moov_at, len: moov.len() as u64 });

        // The caller re-reads that exact range and parses it whole.
        let MoovExtent::Beyond { offset, len } = got else { unreachable!() };
        let re_read = &file[offset as usize..(offset + len) as usize];
        assert_eq!(tags_of(re_read).get("TITLE"), Some("Tail"));

        // With a 64-octet prefix — `ftyp` plus the `mdat` header, the realistic scanner read —
        // the `moov` header itself is out of reach, so the range is the unread tail instead.
        assert_eq!(
            locate_moov(&file[..64], file_len),
            MoovExtent::Beyond { offset: moov_at, len: file_len - moov_at }
        );
    }

    /// When the walk runs off the end of the prefix, the reported range is the unread tail —
    /// which may hold more than the `moov`. Handing those bytes back to `locate_moov` is the
    /// documented protocol, and it terminates: the second answer is `InPrefix`.
    #[test]
    fn locate_moov_unread_tail_is_re_locatable() {
        let mdat = bx(b"mdat", &vec![0u8; 4096]);
        let moov = moov_with_items(&[item(b"\xA9nam", TYPE_UTF8, b"Loop")]);
        let free = bx(b"free", &[0u8; 32]);
        let mut file = mdat.clone();
        file.extend_from_slice(&moov);
        file.extend_from_slice(&free);
        let file_len = file.len() as u64;

        let tail_at = mdat.len() as u64;
        assert_eq!(
            locate_moov(&file[..16], file_len),
            MoovExtent::Beyond { offset: tail_at, len: file_len - tail_at },
            "the tail is `moov` + `free`, not just the box"
        );

        // Second pass over the re-read tail: now the box is fully in hand.
        let tail = &file[tail_at as usize..];
        assert_eq!(
            locate_moov(tail, tail.len() as u64),
            MoovExtent::InPrefix { offset: 0, len: moov.len() }
        );
        assert_eq!(tags_of(&tail[..moov.len()]).get("TITLE"), Some("Loop"));
    }

    /// A `moov` that *starts* in the prefix but is cut short by it is `Beyond`, not `InPrefix`.
    #[test]
    fn locate_moov_cut_short_by_the_prefix_is_beyond() {
        let ftyp = bx(b"ftyp", b"M4A \0\0\0\0");
        let moov = moov_with_items(&[item(b"\xA9nam", TYPE_UTF8, b"Cut")]);
        let mut file = ftyp.clone();
        file.extend_from_slice(&moov);
        let file_len = file.len() as u64;
        let prefix = &file[..ftyp.len() + 12]; // header of `moov` + a few octets
        assert_eq!(
            locate_moov(prefix, file_len),
            MoovExtent::Beyond { offset: ftyp.len() as u64, len: moov.len() as u64 }
        );
    }

    /// A top-level box using the 64-bit `largesize` form (§4.2) is stepped over correctly —
    /// the `mdat > 4 GiB` case, faked here with a small declared size.
    #[test]
    fn locate_moov_honours_64bit_largesize() {
        let payload = [0xAAu8; 32];
        // size == 1, type, largesize (16-octet header), then the body.
        let mut mdat = 1u32.to_be_bytes().to_vec();
        mdat.extend_from_slice(b"mdat");
        mdat.extend_from_slice(&((16 + payload.len()) as u64).to_be_bytes());
        mdat.extend_from_slice(&payload);
        let moov = moov_with_items(&[item(b"\xA9nam", TYPE_UTF8, b"Large")]);
        let mut file = mdat.clone();
        file.extend_from_slice(&moov);
        let file_len = file.len() as u64;

        assert_eq!(
            locate_moov(&file, file_len),
            MoovExtent::InPrefix { offset: mdat.len(), len: moov.len() }
        );
        // With only the `mdat` header in hand, the same walk still reports the right place.
        assert_eq!(
            locate_moov(&file[..16], file_len),
            MoovExtent::Beyond { offset: mdat.len() as u64, len: moov.len() as u64 }
        );
    }

    /// `size == 0` means "to the end of the **file**" (§4.2) — resolved against `file_len`,
    /// not the prefix length, so a size-0 `mdat` correctly swallows the rest of the file and a
    /// size-0 `moov` reports the right length.
    #[test]
    fn locate_moov_resolves_size_zero_against_the_file() {
        let ftyp = bx(b"ftyp", b"isom\0\0\0\0");

        // A size-0 `mdat` runs to EOF: there is no `moov` after it.
        let mut open_mdat = 0u32.to_be_bytes().to_vec();
        open_mdat.extend_from_slice(b"mdat");
        open_mdat.extend_from_slice(&[7u8; 100]);
        let mut file = ftyp.clone();
        file.extend_from_slice(&open_mdat);
        assert_eq!(locate_moov(&file, file.len() as u64), MoovExtent::NotFound);

        // A size-0 `moov` at the end: its length is `file_len - offset`, even when the prefix
        // is short (so the answer is `Beyond`, not a truncated `InPrefix`).
        let moov = moov_with_items(&[item(b"\xA9nam", TYPE_UTF8, b"Open")]);
        let mut open_moov = 0u32.to_be_bytes().to_vec();
        open_moov.extend_from_slice(b"moov");
        open_moov.extend_from_slice(&moov[8..]);
        let mut file = ftyp.clone();
        file.extend_from_slice(&open_moov);
        let file_len = file.len() as u64;
        assert_eq!(
            locate_moov(&file[..ftyp.len() + 10], file_len),
            MoovExtent::Beyond { offset: ftyp.len() as u64, len: open_moov.len() as u64 }
        );
    }

    /// Nothing box-shaped, a header cut in half, and a box longer than the file: all
    /// `NotFound`, none a panic.
    #[test]
    fn locate_moov_rejects_junk_without_panicking() {
        assert_eq!(locate_moov(&[], 0), MoovExtent::NotFound);
        assert_eq!(locate_moov(b"not-an-mp4", 10), MoovExtent::NotFound);
        assert_eq!(locate_moov(&[0, 0, 0, 8, b'f'], 5), MoovExtent::NotFound);
        // A box declaring 4 octets (< its own 8-octet header).
        let mut short = 4u32.to_be_bytes().to_vec();
        short.extend_from_slice(b"ftyp");
        assert_eq!(locate_moov(&short, short.len() as u64), MoovExtent::NotFound);
        // A box declaring more than the file holds.
        let mut huge = u32::MAX.to_be_bytes().to_vec();
        huge.extend_from_slice(b"free");
        assert_eq!(locate_moov(&huge, huge.len() as u64), MoovExtent::NotFound);
    }

    // ---------------------------------------------------------------------------------
    // props
    // ---------------------------------------------------------------------------------

    /// `mdhd` payload (§8.4.2) at the requested version.
    fn mdhd_body(version: u8, timescale: u32, duration: u64) -> Vec<u8> {
        let mut b = vec![version, 0, 0, 0];
        if version == 0 {
            b.extend_from_slice(&[0u8; 8]); // creation/modification time
            b.extend_from_slice(&timescale.to_be_bytes());
            b.extend_from_slice(&(duration as u32).to_be_bytes());
        } else {
            b.extend_from_slice(&[0u8; 16]);
            b.extend_from_slice(&timescale.to_be_bytes());
            b.extend_from_slice(&duration.to_be_bytes());
        }
        b.extend_from_slice(&[0x55, 0xC4, 0, 0]); // language + pre_defined
        b
    }

    /// `mvhd` v0 payload (§8.2.2) — only timescale/duration matter here.
    fn mvhd_body(timescale: u32, duration: u32) -> Vec<u8> {
        let mut b = vec![0u8; 4];
        b.extend_from_slice(&[0u8; 8]);
        b.extend_from_slice(&timescale.to_be_bytes());
        b.extend_from_slice(&duration.to_be_bytes());
        b.extend_from_slice(&[0u8; 80]); // rate, volume, matrix, pre_defined, next_track_ID
        b
    }

    /// `hdlr` payload (§8.4.3): version/flags, pre_defined, handler_type, reserved, name.
    fn hdlr_body(handler: &[u8; 4]) -> Vec<u8> {
        let mut b = vec![0u8; 8];
        b.extend_from_slice(handler);
        b.extend_from_slice(&[0u8; 13]);
        b
    }

    /// An `mp4a` `AudioSampleEntry` (§12.2.3) inside a full `stsd`/`stbl`/`minf` chain.
    fn audio_minf(entry: &[u8]) -> Vec<u8> {
        let mut stsd_body = vec![0u8; 4]; // FullBox version/flags
        stsd_body.extend_from_slice(&1u32.to_be_bytes()); // entry_count
        stsd_body.extend_from_slice(entry);
        let stbl = bx(b"stbl", &bx(b"stsd", &stsd_body));
        bx(b"minf", &stbl)
    }

    fn mp4a_entry(rate: u32, channels: u16) -> Vec<u8> {
        let mut ae = vec![0u8; 28];
        ae[16..18].copy_from_slice(&channels.to_be_bytes());
        ae[24..28].copy_from_slice(&(rate << 16).to_be_bytes()); // 16.16 fixed point
        bx(b"mp4a", &ae)
    }

    /// A `moov` with an optional `mvhd` and one track of the given handler.
    fn moov_with_track(
        mvhd: Option<Vec<u8>>,
        handler: &[u8; 4],
        mdhd: &[u8],
        minf: Option<Vec<u8>>,
    ) -> Vec<u8> {
        let mut mdia = bx(b"hdlr", &hdlr_body(handler));
        mdia.extend_from_slice(&bx(b"mdhd", mdhd));
        if let Some(minf) = minf {
            mdia.extend_from_slice(&minf);
        }
        let mut body = mvhd.map(|m| bx(b"mvhd", &m)).unwrap_or_default();
        body.extend_from_slice(&bx(b"trak", &bx(b"mdia", &mdia)));
        bx(b"moov", &body)
    }

    /// The sound track's `mdhd` sets the duration, in both the 32-bit (v0) and 64-bit (v1)
    /// versions, and wins over a disagreeing `mvhd`.
    #[test]
    fn props_duration_from_sound_track_mdhd_v0_and_v1() {
        // v0: 44100 ticks/s, 441000 ticks = 10 s.
        let moov = moov_with_track(
            Some(mvhd_body(1000, 99_000)), // deliberately wrong: mdhd must win
            b"soun",
            &mdhd_body(0, 44_100, 441_000),
            None,
        );
        assert_eq!(props_from_moov(&moov).duration_ns, Some(10_000_000_000));

        // v1: a 64-bit duration at 48 kHz — 5 400 000 000 ticks = 31.25 hours.
        let moov =
            moov_with_track(None, b"soun", &mdhd_body(1, 48_000, 5_400_000_000), None);
        assert_eq!(props_from_moov(&moov).duration_ns, Some(112_500_000_000_000));
    }

    /// No sound track (video only) → the `mvhd` duration (§8.2.2).
    #[test]
    fn props_falls_back_to_mvhd() {
        let moov = moov_with_track(
            Some(mvhd_body(600, 1_800)), // 3 s
            b"vide",
            &mdhd_body(0, 30_000, 90_000),
            None,
        );
        let p = props_from_moov(&moov);
        assert_eq!(p.duration_ns, Some(3_000_000_000));
        assert_eq!((p.sample_rate, p.channels), (None, None), "no sound track, no audio format");

        // A sound track whose `mdhd` declares no duration also falls back.
        let moov = moov_with_track(Some(mvhd_body(600, 1_200)), b"soun", &mdhd_body(0, 44_100, 0), None);
        assert_eq!(props_from_moov(&moov).duration_ns, Some(2_000_000_000));

        // Nothing declares a duration (the fragmented / still-being-written case).
        let moov = moov_with_track(Some(mvhd_body(600, 0)), b"soun", &mdhd_body(0, 44_100, 0), None);
        assert_eq!(props_from_moov(&moov).duration_ns, None);
    }

    /// Rate and channels come off the sound track's first sample entry (§12.2.3).
    #[test]
    fn props_reads_rate_and_channels_from_the_sample_entry() {
        let moov = moov_with_track(
            None,
            b"soun",
            &mdhd_body(0, 48_000, 96_000),
            Some(audio_minf(&mp4a_entry(48_000, 2))),
        );
        let p = props_from_moov(&moov);
        assert_eq!(p.sample_rate, Some(48_000));
        assert_eq!(p.channels, Some(2));
        assert_eq!(p.duration_ns, Some(2_000_000_000));

        // Any audio four-CC works — the AudioSampleEntry prologue is codec-independent. (The
        // 16.16 field tops out at 65535 Hz; a higher rate needs the version-2 sound
        // description tested below.)
        let mut alac = vec![0u8; 28];
        alac[16..18].copy_from_slice(&6u16.to_be_bytes());
        alac[24..28].copy_from_slice(&(44_100u32 << 16).to_be_bytes());
        let moov = moov_with_track(
            None,
            b"soun",
            &mdhd_body(0, 44_100, 44_100),
            Some(audio_minf(&bx(b"alac", &alac))),
        );
        let p = props_from_moov(&moov);
        assert_eq!((p.sample_rate, p.channels), (Some(44_100), Some(6)));
    }

    /// A QuickTime version-2 sound description carries the true rate as an `f64` and the
    /// channel count as a u32 past the 16.16 placeholder.
    #[test]
    fn props_reads_a_version_2_sound_description() {
        let mut e = vec![0u8; 44];
        e[8..10].copy_from_slice(&2u16.to_be_bytes()); // version 2
        e[24..28].copy_from_slice(&(65_536u32 << 16).to_be_bytes()); // placeholder rate
        e[32..40].copy_from_slice(&192_000f64.to_bits().to_be_bytes());
        e[40..44].copy_from_slice(&8u32.to_be_bytes());
        let moov = moov_with_track(
            None,
            b"soun",
            &mdhd_body(0, 192_000, 192_000),
            Some(audio_minf(&bx(b"lpcm", &e))),
        );
        let p = props_from_moov(&moov);
        assert_eq!((p.sample_rate, p.channels), (Some(192_000), Some(8)));
    }

    /// Props over junk: nothing, never a panic.
    #[test]
    fn props_of_junk_is_empty() {
        assert_eq!(props_from_moov(&[]), Mp4Props::default());
        assert_eq!(props_from_moov(&bx(b"ftyp", b"M4A ")), Mp4Props::default());
        assert_eq!(props_from_moov(&bx(b"moov", b"")), Mp4Props::default());
    }

    // ---------------------------------------------------------------------------------
    // Edit lists (§8.6.5 `edts` / §8.6.6 `elst`)
    // ---------------------------------------------------------------------------------

    /// An `elst` payload (§8.6.6) at the requested version, from `(segment_duration,
    /// media_time)` pairs. `media_rate` is written as the 1.0 every real file carries.
    fn elst_body(version: u8, edits: &[(u64, i64)]) -> Vec<u8> {
        let mut b = vec![version, 0, 0, 0];
        b.extend_from_slice(&(edits.len() as u32).to_be_bytes());
        for &(segment, media_time) in edits {
            if version == 1 {
                b.extend_from_slice(&segment.to_be_bytes());
                b.extend_from_slice(&media_time.to_be_bytes());
            } else {
                b.extend_from_slice(&(segment as u32).to_be_bytes());
                b.extend_from_slice(&(media_time as i32).to_be_bytes());
            }
            b.extend_from_slice(&[0, 1, 0, 0]); // media_rate 1.0 (16.16)
        }
        b
    }

    /// A `moov` with an `mvhd` and one sound track carrying an optional `edts`/`elst`.
    fn moov_with_edits(
        movie_timescale: u32,
        movie_duration: u32,
        mdhd: &[u8],
        elst: Option<Vec<u8>>,
    ) -> Vec<u8> {
        let mut mdia = bx(b"hdlr", &hdlr_body(b"soun"));
        mdia.extend_from_slice(&bx(b"mdhd", mdhd));
        let mut trak = Vec::new();
        if let Some(elst) = elst {
            trak.extend_from_slice(&bx(b"edts", &bx(b"elst", &elst)));
        }
        trak.extend_from_slice(&bx(b"mdia", &mdia));
        let mut body = bx(b"mvhd", &mvhd_body(movie_timescale, movie_duration));
        body.extend_from_slice(&bx(b"trak", &trak));
        bx(b"moov", &body)
    }

    /// The case the whole feature exists for: an AAC file whose `mdhd` counts the encoder's
    /// 1024 priming samples and whose edit list trims them (§8.6.6).
    ///
    /// 48 kHz, 480 000 media ticks = 10.000 s of stored samples; the edit presents from
    /// `media_time = 1024` for 9 979 movie ticks (ms) = 9.979 s. The trim is 21 ms — far more
    /// than the 1 ms movie tick — so the edit list is believed over the media header.
    #[test]
    fn elst_trimming_priming_wins_over_mdhd() {
        let moov = moov_with_edits(
            1_000,
            9_979,
            &mdhd_body(0, 48_000, 480_000),
            Some(elst_body(0, &[(9_979, 1_024)])),
        );
        assert_eq!(props_from_moov(&moov).duration_ns, Some(9_979_000_000));
    }

    /// A leading **empty edit** (`media_time == -1`, §8.6.6) is a gap: it delays the start but
    /// presents no media, so only the non-empty edits are summed.
    ///
    /// 500 ms of silence, then 9 979 ms of media → 9.979 s presented, not 10.479 s.
    #[test]
    fn elst_leading_empty_edit_contributes_no_duration() {
        let moov = moov_with_edits(
            1_000,
            10_479,
            &mdhd_body(0, 48_000, 480_000),
            Some(elst_body(0, &[(500, -1), (9_979, 1_024)])),
        );
        assert_eq!(props_from_moov(&moov).duration_ns, Some(9_979_000_000));

        // An edit list of *nothing but* empty edits presents no media at all, so it states no
        // duration and the media header stands.
        let moov = moov_with_edits(
            1_000,
            500,
            &mdhd_body(0, 48_000, 480_000),
            Some(elst_body(0, &[(500, -1)])),
        );
        assert_eq!(props_from_moov(&moov).duration_ns, Some(10_000_000_000));
    }

    /// Version 1 widens `segment_duration` to 64 bits and `media_time` to a signed 64
    /// (§8.6.6). Both the trim and the empty-edit sentinel must read at that width.
    #[test]
    fn elst_version_1_reads_64_bit_fields() {
        // A duration past u32::MAX movie ticks: 5 000 000 000 ms ≈ 57.9 days.
        let moov = moov_with_edits(
            1_000,
            0,
            &mdhd_body(1, 48_000, 240_000_000_000),
            Some(elst_body(1, &[(5_000_000_000, 1_024)])),
        );
        assert_eq!(props_from_moov(&moov).duration_ns, Some(5_000_000_000_000_000));

        // `-1` at 64 bits is a different bit pattern from `-1` at 32; the empty-edit test must
        // still fire, leaving the media header (240e9 / 48 kHz = 5 000 000 s) in place.
        let moov = moov_with_edits(
            1_000,
            0,
            &mdhd_body(1, 48_000, 240_000_000_000),
            Some(elst_body(1, &[(9_999, -1)])),
        );
        assert_eq!(props_from_moov(&moov).duration_ns, Some(5_000_000_000_000_000));
    }

    /// The common, *harmless* edit list: one `media_time == 0` entry that trims nothing and
    /// merely restates the media duration rounded up into the coarse movie timescale.
    ///
    /// These are the four files in the reference library that a naive "always apply the elst"
    /// rule puts 0.17–0.87 ms off `ffprobe`. 44 100 ticks/s, 15 129 600 ticks = 343.074 830 s;
    /// the `elst` rounds that up to 343 075 ms. The difference is under one movie tick, so the
    /// finer media header is kept. (Real numbers, from `Carbon_Based_Lifeforms-Accede.m4a`.)
    #[test]
    fn elst_that_only_restates_the_media_duration_keeps_the_finer_mdhd() {
        let moov = moov_with_edits(
            1_000,
            343_075,
            &mdhd_body(0, 44_100, 15_129_600),
            Some(elst_body(0, &[(343_075, 0)])),
        );
        // 15_129_601 / 44_100 s exactly, in ns — *not* the 343_075_000_000 the elst rounds to.
        assert_eq!(props_from_moov(&moov).duration_ns, Some(343_074_829_931));
    }

    /// With no `edts` at all, the media header is reported unchanged — the pre-existing
    /// behaviour every file without an edit list relies on.
    #[test]
    fn absent_elst_leaves_the_duration_unchanged() {
        let with = moov_with_edits(1_000, 10_000, &mdhd_body(0, 48_000, 480_000), None);
        assert_eq!(props_from_moov(&with).duration_ns, Some(10_000_000_000));

        // An `edts` with no `elst` inside it, and an `elst` declaring zero entries, both say
        // nothing rather than saying zero.
        let empty_edts = {
            let mut mdia = bx(b"hdlr", &hdlr_body(b"soun"));
            mdia.extend_from_slice(&bx(b"mdhd", &mdhd_body(0, 48_000, 480_000)));
            let mut trak = bx(b"edts", b"");
            trak.extend_from_slice(&bx(b"mdia", &mdia));
            bx(b"moov", &bx(b"trak", &trak))
        };
        assert_eq!(props_from_moov(&empty_edts).duration_ns, Some(10_000_000_000));

        let no_entries =
            moov_with_edits(1_000, 10_000, &mdhd_body(0, 48_000, 480_000), Some(elst_body(0, &[])));
        assert_eq!(props_from_moov(&no_entries).duration_ns, Some(10_000_000_000));
    }

    /// A hostile `elst` states nothing rather than a wrong duration, and never panics: an
    /// unknown version (unknown entry width), an `entry_count` the box cannot carry, and a
    /// segment-duration sum that overflows `u64` all fall back to the media header.
    #[test]
    fn hostile_elst_falls_back_to_the_media_header() {
        let mdhd = mdhd_body(0, 48_000, 480_000);
        let ten_s = Some(10_000_000_000);

        // Version 7: this parser cannot know how wide an entry is, so it walks none of them.
        let mut v7 = elst_body(0, &[(9_979, 1_024)]);
        v7[0] = 7;
        assert_eq!(props_from_moov(&moov_with_edits(1_000, 0, &mdhd, Some(v7))).duration_ns, ten_s);

        // `entry_count` claims 2^32-1 entries in a box holding one.
        let mut liar = elst_body(0, &[(9_979, 1_024)]);
        liar[4..8].copy_from_slice(&u32::MAX.to_be_bytes());
        assert_eq!(
            props_from_moov(&moov_with_edits(1_000, 0, &mdhd, Some(liar))).duration_ns,
            ten_s
        );

        // Two v1 edits whose durations sum past u64::MAX.
        let overflow = elst_body(1, &[(u64::MAX, 0), (u64::MAX, 0)]);
        assert_eq!(
            props_from_moov(&moov_with_edits(1_000, 0, &mdhd, Some(overflow))).duration_ns,
            ten_s
        );

        // Every truncation of an edit-list fixture parses without panicking.
        let full = moov_with_edits(
            1_000,
            9_979,
            &mdhd,
            Some(elst_body(0, &[(500, -1), (9_979, 1_024), (7, 3)])),
        );
        for n in 0..full.len() {
            let _ = props_from_moov(&full[..n]);
        }
    }

    /// A movie whose **video** track outlasts its sound track reports the movie duration
    /// (§8.2.2), not the audio's — the file's playing time is the longest track's.
    ///
    /// The numbers are `~/Music/ffmpeg_bar_test/output.mp4`: a 19 200-tick video track running
    /// 427.733 s beside a 48 kHz sound track whose edit list presents 427.731 s. `ffprobe`
    /// reports 427.733333 for the format and 427.731000 for the audio stream.
    #[test]
    fn a_longer_video_track_sets_the_movie_duration() {
        let mut sound = bx(b"edts", &bx(b"elst", &elst_body(0, &[(427_731, 688)])));
        let mut mdia = bx(b"hdlr", &hdlr_body(b"soun"));
        mdia.extend_from_slice(&bx(b"mdhd", &mdhd_body(0, 48_000, 20_531_761)));
        sound.extend_from_slice(&bx(b"mdia", &mdia));

        let mut video = bx(b"edts", &bx(b"elst", &elst_body(0, &[(427_734, 2_048)])));
        let mut vmdia = bx(b"hdlr", &hdlr_body(b"vide"));
        vmdia.extend_from_slice(&bx(b"mdhd", &mdhd_body(0, 19_200, 8_212_480)));
        video.extend_from_slice(&bx(b"mdia", &vmdia));

        let mut body = bx(b"mvhd", &mvhd_body(1_000, 427_734));
        body.extend_from_slice(&bx(b"trak", &video));
        body.extend_from_slice(&bx(b"trak", &sound));
        let moov = bx(b"moov", &body);

        // 427.734 s — the movie header — within a millisecond of ffprobe's 427.733333, where
        // the raw sound `mdhd` would say 427.745 and its edit list 427.731.
        assert_eq!(props_from_moov(&moov).duration_ns, Some(427_734_000_000));
    }

    /// …but in a **single-track** movie the same excess is a wrong movie header, not a longer
    /// track, so the sound track's own duration stands (§8.2.2 defines `mvhd` as the longest
    /// track's — with one track, that is this one).
    #[test]
    fn a_lone_sound_track_outranks_a_disagreeing_movie_header() {
        let moov = moov_with_edits(1_000, 99_000, &mdhd_body(0, 48_000, 480_000), None);
        assert_eq!(props_from_moov(&moov).duration_ns, Some(10_000_000_000));
    }

    // ---------------------------------------------------------------------------------
    // Untrusted input
    // ---------------------------------------------------------------------------------

    /// Every truncation of a full fixture — tags *and* props *and* the locator — parses without
    /// panicking (spec: "a crash on bad input is a P0"). The prefix walk is exercised at every
    /// cut too, which is exactly what a scanner reading a partial file does.
    #[test]
    fn truncation_never_panics() {
        let mut ff = full_bx(b"mean", b"com.apple.iTunes");
        ff.extend_from_slice(&full_bx(b"name", b"replaygain_track_gain"));
        ff.extend_from_slice(&data_box(TYPE_UTF8, b"-7.35 dB"));
        let mut trkn = vec![0u8, 0];
        trkn.extend_from_slice(&3u16.to_be_bytes());
        trkn.extend_from_slice(&12u16.to_be_bytes());
        trkn.extend_from_slice(&[0, 0]);

        // A `moov` carrying both metadata and a full sound track.
        let items = [
            item(b"\xA9nam", TYPE_UTF8, "Enough".as_bytes()),
            item(b"trkn", TYPE_IMPLICIT, &trkn),
            item(b"covr", TYPE_PNG, b"\x89PNG\r\n\x1a\n"),
            bx(b"----", &ff),
        ];
        let mut ilst = Vec::new();
        for i in &items {
            ilst.extend_from_slice(i);
        }
        let mut meta_body = bx(b"hdlr", &hdlr_body(b"mdir"));
        meta_body.extend_from_slice(&bx(b"ilst", &ilst));
        let udta = bx(b"udta", &full_bx(b"meta", &meta_body));

        let mut mdia = bx(b"hdlr", &hdlr_body(b"soun"));
        mdia.extend_from_slice(&bx(b"mdhd", &mdhd_body(0, 44_100, 441_000)));
        mdia.extend_from_slice(&audio_minf(&mp4a_entry(44_100, 2)));
        let mut moov_body = bx(b"mvhd", &mvhd_body(600, 6_000));
        moov_body.extend_from_slice(&bx(b"trak", &bx(b"mdia", &mdia)));
        moov_body.extend_from_slice(&udta);
        let moov = bx(b"moov", &moov_body);

        // The complete fixture is sane first, so the loop below is truncating something real.
        let t = tags_of(&moov);
        assert_eq!(t.get("TITLE"), Some("Enough"));
        assert_eq!(t.get("TRACKNUMBER"), Some("3/12"));
        assert_eq!(t.get("REPLAYGAIN_TRACK_GAIN"), Some("-7.35 dB"));
        assert_eq!(t.pics.len(), 1);
        let p = props_from_moov(&moov);
        assert_eq!(p.duration_ns, Some(10_000_000_000));
        assert_eq!((p.sample_rate, p.channels), (Some(44_100), Some(2)));

        let mut file = bx(b"ftyp", b"M4A \0\0\0\0");
        file.extend_from_slice(&moov);
        for n in 0..=file.len() {
            let _ = locate_moov(&file[..n], file.len() as u64);
        }
        for n in 0..=moov.len() {
            let cut = &moov[..n];
            let _ = tags_of(cut);
            let _ = props_from_moov(cut);
        }

        // And every truncation with the outer `moov` size left *intact* (the nastier case: the
        // header promises bytes that are not there).
        for n in 8..moov.len() {
            let mut cut = moov[..n].to_vec();
            cut[..4].copy_from_slice(&(moov.len() as u32).to_be_bytes());
            let _ = tags_of(&cut);
            let _ = props_from_moov(&cut);
        }
    }

    /// A stack-formatted number that would overflow its buffer is dropped, not truncated.
    #[test]
    fn fixed_str_overflow_is_sticky() {
        let mut s = FixedStr::<4>::new();
        assert!(write!(s, "12").is_ok());
        assert_eq!(s.as_str(), Some("12"));
        assert!(write!(s, "34567").is_err());
        assert_eq!(s.as_str(), None, "an overflowed buffer yields nothing at all");
    }
}
