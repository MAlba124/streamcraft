//! `\Segment\Tags` and `\Segment\Attachments` — the metadata a Matroska file carries about
//! itself, parsed straight out of a borrowed byte window (spec: `spec/MATROSKA.md §tags`;
//! RFC 9559 §5.1.7 Attachments, §5.1.8 Tags).
//!
//! This is the read-only half of the container a *scanner* wants: no demuxer, no reader state
//! machine, no allocation. Everything here is a pure function over `&[u8]` that emits into a
//! [`TagSink`], so a caller that has already read some bytes — a 128 KiB prefix, a positioned
//! extent — parses them where they lie.
//!
//! ## The problem this module actually solves: where the metadata *is*
//!
//! A Matroska Segment is a flat sequence of level-1 masters, and the interesting ones are at
//! opposite ends of the file. `Info` (which carries the duration) sits at the front, before the
//! first Cluster. `Tags` and `Attachments` are allowed anywhere and in practice are written
//! **after** the Clusters — past a hundred megabytes of video — because a single-pass muxer
//! does not know them until it is done. Scanning backwards from EOF would find them, but
//! Matroska has no trailer and no back-pointer, so that is a guess.
//!
//! The structural answer is the `SeekHead` (§5.1.1): an index at the *start* of the Segment
//! mapping each level-1 master's Element ID to its Segment Position. [`index_segment`] reads it
//! out of the prefix a scanner has already paid for and reports absolute file offsets for
//! `Info`, `Tags` and `Attachments` — so reaching a tag block at the end of an 80 MB file costs
//! one positioned read, not a scan. Files whose masters sit inside the prefix are found by the
//! same walk without any SeekHead at all.
//!
//! A `SeekHead` states a *position* but not a *length*, so the element's own header at that
//! offset is what tells the caller how much to read; [`element_extent`] answers that from
//! whatever bytes came back, and says so when the read was short. That is the same
//! read-then-ask-again protocol `pf_mp4::ilst::locate_moov` uses for a trailing `moov`.
//!
//! ## Scope: which tags describe *this file*
//!
//! A `Tag` is not unconditionally about the file. Its `Targets` (§5.1.8.1.1) name a logical
//! level — 70 COLLECTION, 60 EDITION, 50 ALBUM, 40 PART, 30 TRACK, 20 SUBTRACK, 10 SHOT,
//! **defaulting to 50** — and may pin the tag to one chapter, edition or attachment by UID.
//! [`parse_tags`] emits the levels that describe the file or its tracks (30 and up) and drops
//! the ones that describe a fragment of it (20 SUBTRACK, 10 SHOT: a tag on one movement of a
//! recording is not the recording's title), along with anything scoped to a chapter, edition or
//! attachment UID. See [`emits_for_file`].
//!
//! ## Allocation
//!
//! None. Names and values reach the sink borrowed from the caller's buffer, cover art included
//! — an attached JPEG is handed over as a sub-slice of the read buffer and never copied here
//! (spec: allocation discipline). The one transformation, folding a lower-case tag name to
//! upper case, writes into a stack buffer; a name too long for it passes through unchanged.
//!
//! ## Hostile input
//!
//! Every walk is bounds-checked and every recursion depth-capped ([`MAX_NEST`]); a malformed
//! child ends the walk that contains it rather than failing the file, so a corrupt `Tags`
//! element costs the tags after the corruption and nothing else. Parsing is total: fewer tags,
//! never a panic (spec: "a crash on bad input is a P0").

use profluens_core::event::TagSink;

use crate::ebml::{self, id};
use crate::reader::{parse_info, read_uint};

/// How deep `SimpleTag` nesting is followed (§5.1.8.1.2 makes it unbounded). Real files nest
/// one level — `TITLE` with a `SORT_WITH` under it — and a cap turns a crafted file's
/// thousand-deep chain into a bounded walk rather than a blown stack.
pub const MAX_NEST: usize = 6;

/// `Targets\TargetTypeValue` at or above which a tag is taken to describe the file or its
/// tracks: 30 is TRACK / SONG / CHAPTER (§5.1.8.1.1.1, Table 33). Below it are SUBTRACK (20)
/// and SHOT (10), which describe a *part* of a track.
pub const TARGET_TRACK: u64 = 30;

/// The default `TargetTypeValue` when `Targets` omits it — 50, ALBUM (§5.1.8.1.1.1).
pub const TARGET_DEFAULT: u64 = 50;

/// The longest tag name folded to upper case in place. Past this the name is passed through as
/// written, which costs nothing: Matroska tag names are upper case by convention already
/// (§5.1.8.1.2.1, [MatroskaTags]), so the fold is a fallback, not the common path.
const MAX_KEY: usize = 64;

// =====================================================================================
// 1. Locating the metadata masters
// =====================================================================================

/// Absolute file offsets of a Segment's metadata masters, as far as a prefix reveals them.
///
/// Each is the offset of the element's **ID octet** — the first byte of its header — so a
/// caller reads from there and hands the bytes to [`element_extent`] to learn how far the
/// element runs. `None` means the prefix neither contained the element nor found it in a
/// `SeekHead`; for `Tags` and `Attachments` that usually means the file has none.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct SegmentIndex {
    /// Absolute offset of the first byte of the Segment's *data* — the base every Segment
    /// Position in a `SeekHead` is relative to (RFC 9559 §4).
    pub segment_data_start: u64,
    /// `\Segment\Info` (§5.1.2) — the TimestampScale and Duration.
    pub info: Option<u64>,
    /// `\Segment\Tracks` (§5.1.4) — the per-track sample rate and channel count.
    pub tracks: Option<u64>,
    /// `\Segment\Tags` (§5.1.8).
    pub tags: Option<u64>,
    /// `\Segment\Attachments` (§5.1.7).
    pub attachments: Option<u64>,
}

impl SegmentIndex {
    /// The offsets this index found, in ascending order, for a caller deciding what to read.
    /// Handy because `Tags` and `Attachments` are usually adjacent at the end of the file, so
    /// one read spanning the lowest of them onwards picks up both.
    pub fn lowest(&self) -> Option<u64> {
        [self.info, self.tracks, self.tags, self.attachments].into_iter().flatten().min()
    }
}

/// Walk a file `prefix` — which must begin at **byte 0** — and report where the Segment's
/// metadata masters are (see [`SegmentIndex`]).
///
/// Two sources, in this order of preference: the level-1 masters actually present in the
/// window (walked until the first Cluster, where frames begin), and the front `SeekHead`'s
/// index of the ones that are not (§5.1.1). A position found in the window is trusted over a
/// `SeekHead` entry, because it is the element rather than a claim about it.
///
/// A `SeekHead` may point at a *second* `SeekHead` rather than at the masters directly
/// (§5.1.1 permits the chain, and a file that appends metadata after muxing uses it). One hop
/// is followed when the linked SeekHead is inside the window; a chain that leaves the prefix is
/// reported as far as it got rather than followed, since following it is a read the caller has
/// to make, not this function.
///
/// `None` only when the Segment itself cannot be located: not a Matroska file, or a prefix
/// truncated before the Segment header.
pub fn index_segment(prefix: &[u8]) -> Option<SegmentIndex> {
    // The Segment's data start (RFC 9559 §4): a linear top-level walk stepping over the
    // definite-size EBML Header, exactly as `parse_seek_head` does.
    let mut at = 0usize;
    let segment_data_start = loop {
        let h = ebml::read_element_header(prefix, at).ok()?;
        if h.id == id::SEGMENT {
            // An unknown-size streamed master: its data starts right past the 0xFF marker.
            break h.data_start as u64;
        }
        let end = h.data_end().ok()??;
        if end <= at || end > prefix.len() {
            return None; // truncated before the Segment, or a malformed size
        }
        at = end;
    };

    let mut index = SegmentIndex { segment_data_start, ..SegmentIndex::default() };
    let mut seek_heads = 0usize;
    let mut at = segment_data_start as usize;
    // A second SeekHead discovered mid-walk is visited by restarting the level-1 scan at it;
    // `seek_heads` bounds that to one hop so a self-referential file cannot loop.
    while at < prefix.len() {
        let Ok(h) = ebml::read_element_header(prefix, at) else { break };
        if h.id == id::CLUSTER {
            break; // frames begin; the front metadata is behind us
        }
        // Every level-1 master before the first Cluster is definite-size. One that runs past
        // the window means the caller holds too little prefix — keep what we have.
        let Ok(Some(end)) = h.data_end() else { break };
        if end > prefix.len() || end <= at {
            break;
        }
        // The element is here, in full: its own offset beats anything a SeekHead claims.
        match h.id {
            i if i == id::INFO => index.info = Some(at as u64),
            i if i == id::TRACKS => index.tracks = Some(at as u64),
            i if i == id::TAGS => index.tags = Some(at as u64),
            i if i == id::ATTACHMENTS => index.attachments = Some(at as u64),
            i if i == id::SEEK_HEAD && seek_heads < 2 => {
                seek_heads += 1;
                index_seek_head(&prefix[h.data_start..end], segment_data_start, &mut index);
            }
            _ => {}
        }
        at = end;
    }
    Some(index)
}

/// Fold one `SeekHead` master's entries into `index`, rebasing each Segment Position onto an
/// absolute file offset (RFC 9559 §5.1.1, §4). Entries already answered by an element found in
/// the window are left alone. Malformed entries are skipped, not fatal.
fn index_seek_head(data: &[u8], segment_data_start: u64, index: &mut SegmentIndex) {
    let mut at = 0usize;
    while at < data.len() {
        let Ok(h) = ebml::read_element_header(data, at) else { return };
        let Ok(Some(end)) = h.data_end() else { return };
        if end > data.len() || end <= at {
            return;
        }
        if h.id == id::SEEK {
            if let Some((target, pos)) = seek_entry(&data[h.data_start..end]) {
                let abs = segment_data_start.saturating_add(pos);
                // The raw ID octets are compared verbatim: a SeekID's payload *is* the target
                // element's ID, length-descriptor bits included (§5.1.1).
                match target {
                    t if t == id::INFO => index.info.get_or_insert(abs),
                    t if t == id::TRACKS => index.tracks.get_or_insert(abs),
                    t if t == id::TAGS => index.tags.get_or_insert(abs),
                    t if t == id::ATTACHMENTS => index.attachments.get_or_insert(abs),
                    _ => &mut 0,
                };
            }
        }
        at = end;
    }
}

/// One `Seek` entry as `(SeekID payload, SeekPosition)` (§5.1.1). `None` when either child is
/// missing or the entry is malformed.
fn seek_entry(data: &[u8]) -> Option<(&[u8], u64)> {
    let mut at = 0usize;
    let mut target = None;
    let mut pos = None;
    while at < data.len() {
        let h = ebml::read_element_header(data, at).ok()?;
        let end = h.data_end().ok()??;
        if end > data.len() || end <= at {
            return None;
        }
        let body = &data[h.data_start..end];
        if h.id == id::SEEK_ID {
            target = Some(body);
        } else if h.id == id::SEEK_POSITION {
            pos = Some(read_uint(body));
        }
        at = end;
    }
    Some((target?, pos?))
}

/// What a window holds of the element whose header begins at `window[0]`.
///
/// This is the answer to "the `SeekHead` gave me a position, how much do I read?": issue a read
/// of whatever size is convenient, hand the bytes here, and either the element is complete or
/// the reply names the length that would complete it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Extent {
    /// The complete element — header included — is `window[..len]`.
    Complete { len: usize },
    /// The element is `total` bytes long and the window holds fewer. Read `total` bytes from
    /// the same offset and ask again.
    Short { total: u64 },
    /// No element header could be read: the window is shorter than one, or the bytes at this
    /// offset are not an EBML element (a stale or hostile `SeekHead` position).
    Invalid,
}

/// Measure the element whose header begins at `window[0]` (RFC 8794 §4–§6). See [`Extent`].
///
/// An unknown-size master (§6.2) is [`Extent::Invalid`] here: it has no declared end, so there
/// is no length to read. Only `Segment` and `Cluster` are written that way in practice, and
/// neither is a metadata master a caller looks up.
pub fn element_extent(window: &[u8]) -> Extent {
    let Ok(h) = ebml::read_element_header(window, 0) else { return Extent::Invalid };
    let Ok(Some(end)) = h.data_end() else { return Extent::Invalid };
    if end <= window.len() {
        Extent::Complete { len: end }
    } else {
        Extent::Short { total: end as u64 }
    }
}

/// The `\Segment\Info` duration in nanoseconds (RFC 9559 §5.1.2): `Duration` — a float in
/// TimestampScale ticks — times `TimestampScale`, the ns per tick.
///
/// `info` is the Info element **including** its ID + size header, i.e. exactly the bytes at the
/// offset [`SegmentIndex::info`] reports. `None` when the element declares no duration (a live
/// or still-being-written stream), or is malformed.
///
/// The tick arithmetic is [`crate::reader`]'s, reused rather than re-derived: the streaming
/// reader converts the same two fields the same way for `duration_ns`.
pub fn info_duration_ns(info: &[u8]) -> Option<u64> {
    let h = ebml::read_element_header(info, 0).ok()?;
    if h.id != id::INFO {
        return None;
    }
    let end = h.data_end().ok()??;
    let body = info.get(h.data_start..end.min(info.len()))?;
    let (scale, ticks) = parse_info(body).ok()?;
    let scale = scale.unwrap_or(crate::writer::DEFAULT_TIMESTAMP_SCALE);
    let ticks = ticks?;
    // A negative, non-finite or absurd duration is a corrupt field, not a length.
    let ns = ticks * scale as f64;
    (ns.is_finite() && ns >= 0.0 && ns < u64::MAX as f64).then(|| ns.round() as u64)
}

/// `(sample_rate, channels)` from the first **audio** track of a `\Segment\Tracks` element
/// (RFC 9559 §5.1.4): the `TrackEntry` whose `TrackType` is 2, and its `Audio` master's
/// `SamplingFrequency` (a float, §5.1.4.1.28.1) and `Channels` (§5.1.4.1.28.2).
///
/// `tracks` is the element **including** its ID + size header, as [`SegmentIndex`] reports it.
/// `None` when the file declares no audio track; either field may still be zero, which the
/// caller reads as "not stated" — `Channels` defaults to 1 and `SamplingFrequency` to 8000 in
/// the spec, but a scanner reporting a guessed rate as fact is worse than reporting none.
///
/// `OutputSamplingFrequency` (the rate after SBR/PS upsampling) is deliberately ignored: it
/// describes what a decoder emits, and every other format in the scanner reports the rate the
/// stream is *stored* at.
pub fn audio_format(tracks: &[u8]) -> Option<(u32, u32)> {
    let body = master_body(tracks, id::TRACKS)?;
    let mut found = None;
    for_each_child(body, |h, entry| {
        if h.id != id::TRACK_ENTRY || found.is_some() {
            return;
        }
        let mut is_audio = false;
        let mut rate = 0u32;
        let mut channels = 0u32;
        for_each_child(entry, |c, cbody| match c.id {
            // TrackType 2 = audio (§5.1.4.1.3).
            i if i == id::TRACK_TYPE => is_audio = read_uint(cbody) == 2,
            i if i == id::AUDIO => {
                for_each_child(cbody, |a, abody| match a.id {
                    j if j == id::SAMPLING_FREQUENCY => {
                        // An EBML float is 4 or 8 octets (RFC 8794 §7.3); a rate that is not a
                        // sane positive number is a corrupt field, not a format.
                        let hz = match abody.len() {
                            4 => f32::from_bits(u32::from_be_bytes(
                                abody.try_into().unwrap_or_default(),
                            )) as f64,
                            8 => f64::from_bits(u64::from_be_bytes(
                                abody.try_into().unwrap_or_default(),
                            )),
                            _ => 0.0,
                        };
                        if hz.is_finite() && hz > 0.0 && hz < u32::MAX as f64 {
                            rate = hz as u32;
                        }
                    }
                    j if j == id::CHANNELS => channels = read_uint(abody).min(u32::MAX as u64) as u32,
                    _ => {}
                });
            }
            _ => {}
        });
        if is_audio {
            found = Some((rate, channels));
        }
    });
    found
}

// =====================================================================================
// 2. Tags
// =====================================================================================

/// Parse a `\Segment\Tags` element (RFC 9559 §5.1.8) into `sink`.
///
/// `tags` is the element **including** its ID + size header — the bytes at the offset
/// [`SegmentIndex::tags`] reports. A window holding more than the element is fine (the declared
/// size bounds the walk); one holding less is parsed as far as it goes.
///
/// ## Which tags are emitted
///
/// Only those whose `Targets` describe the file or its tracks — see [`emits_for_file`].
///
/// ## How names are mapped
///
/// Matroska tag names are already the upper-case, Vorbis-adjacent vocabulary the rest of this
/// workspace keys on ([MatroskaTags]), so most pass straight through. The exceptions are the
/// names Matroska spells differently, and the two that mean different things at different
/// levels:
///
/// | Matroska | emitted as |
/// |---|---|
/// | `DATE_RELEASED`, `DATE_RECORDED`, `DATE_WRITTEN` | `DATE` |
/// | `PART_NUMBER` | `TRACKNUMBER` — or `DISCNUMBER` at album level |
/// | `TOTAL_PARTS` | `TRACKTOTAL` — or `DISCTOTAL` at album level |
/// | `TITLE` | `TITLE` — or `ALBUM` at album level |
/// | `ARTIST` | `ARTIST` — or `ALBUMARTIST` at album level |
/// | anything else | itself, folded to upper case |
///
/// "**At album level**" is the subtle one. A `Tag` at level 50 means *the album's* title, and a
/// file that carries both a level-50 and a level-30 block is saying "album X, track Y" — so
/// emitting both as `TITLE` would let the album name win the key. But the overwhelmingly common
/// file has exactly one block and omits `TargetTypeValue` entirely, which *defaults* to 50
/// while plainly meaning "this recording": reading that as an album title would leave the file
/// with no title at all. So the remap is conditional on the file actually distinguishing the
/// two — it applies only when some `Tag` in the element declares a level below 50. See
/// [`has_track_level`].
///
/// Nested `SimpleTag`s (§5.1.8.1.2) are flattened into the same flat key space, parent before
/// child so a first-value-wins sink keeps the outer one. Matroska nests to qualify a tag
/// (`TITLE` → `SORT_WITH`); with a Vorbis-comment-shaped sink there is nowhere to put the
/// qualification, and dropping the children outright would lose more than flattening them.
pub fn parse_tags(tags: &[u8], sink: &mut impl TagSink) {
    let Some(body) = master_body(tags, id::TAGS) else { return };
    let album_scope = has_track_level(body);
    for_each_child(body, |h, tag| {
        if h.id == id::TAG {
            parse_tag(tag, album_scope, sink);
        }
    });
}

/// Whether any `Tag` in a `Tags` body declares a level below [`TARGET_DEFAULT`] — i.e. whether
/// the file distinguishes track-level metadata from album-level metadata at all. See
/// [`parse_tags`] for why the name mapping turns on this.
fn has_track_level(body: &[u8]) -> bool {
    let mut found = false;
    for_each_child(body, |h, tag| {
        if h.id == id::TAG && target_of(tag).level < TARGET_DEFAULT {
            found = true;
        }
    });
    found
}

/// A `Targets` master reduced to what decides whether — and how — its `Tag` is emitted.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Target {
    /// `TargetTypeValue`, defaulted to [`TARGET_DEFAULT`] when absent (§5.1.8.1.1.1).
    level: u64,
    /// Whether the tag is pinned to a specific chapter, edition or attachment by UID
    /// (§5.1.8.1.1.4–6) — i.e. describes a part of the file rather than the file.
    pinned: bool,
}

impl Default for Target {
    fn default() -> Self {
        Self { level: TARGET_DEFAULT, pinned: false }
    }
}

/// Whether a `Tag` at this target describes the file or its tracks, and so belongs in a
/// scanner's output.
///
/// Two ways to fail it (RFC 9559 §5.1.8.1.1). A level below [`TARGET_TRACK`] — SUBTRACK (20) or
/// SHOT (10) — describes a movement or a scene, not the recording. And a non-zero
/// `TagChapterUID`, `TagEditionUID` or `TagAttachmentUID` pins the tag to one chapter, edition
/// or attached file; a chapter's title is not the file's title. `TagTrackUID` is deliberately
/// *not* a disqualifier: it names which track a tag applies to, and a scanner reporting one
/// file's metadata wants the track's tags whether or not they are scoped to it.
pub fn emits_for_file(level: u64, pinned: bool) -> bool {
    level >= TARGET_TRACK && !pinned
}

/// Read a `Tag`'s `Targets` child (§5.1.8.1.1). An absent `Targets` is the default target,
/// which is exactly what a file with one untargeted tag block means.
fn target_of(tag: &[u8]) -> Target {
    let mut target = Target::default();
    for_each_child(tag, |h, targets| {
        if h.id != id::TARGETS {
            return;
        }
        for_each_child(targets, |c, body| match c.id {
            i if i == id::TARGET_TYPE_VALUE => target.level = read_uint(body),
            i if i == id::TAG_CHAPTER_UID
                || i == id::TAG_EDITION_UID
                || i == id::TAG_ATTACHMENT_UID =>
            {
                // Each defaults to 0 = "not scoped"; only a non-zero UID pins the tag.
                target.pinned |= read_uint(body) != 0;
            }
            _ => {}
        });
    });
    target
}

/// Emit one `Tag`'s `SimpleTag`s, if its target says it describes the file.
fn parse_tag(tag: &[u8], album_scope: bool, sink: &mut impl TagSink) {
    let target = target_of(tag);
    if !emits_for_file(target.level, target.pinned) {
        return;
    }
    // The album remap applies to the levels that *group* recordings — ALBUM (50) and up — and
    // only when the file has a track level to distinguish them from (see `parse_tags`).
    let album = album_scope && target.level >= TARGET_DEFAULT;
    for_each_child(tag, |h, simple| {
        if h.id == id::SIMPLE_TAG {
            emit_simple_tag(simple, album, 0, sink);
        }
    });
}

/// Emit one `SimpleTag` (§5.1.8.1.2) and, depth permitting, the ones nested inside it.
fn emit_simple_tag(simple: &[u8], album: bool, depth: usize, sink: &mut impl TagSink) {
    let mut name: Option<&str> = None;
    let mut value: Option<&str> = None;
    for_each_child(simple, |h, body| match h.id {
        // A tag name or string that is not valid UTF-8 violates §5.1.8.1.2.1/§5.1.8.1.2.5's
        // `utf-8` type; the pair is dropped rather than guessed at.
        i if i == id::TAG_NAME => name = core::str::from_utf8(body).ok(),
        i if i == id::TAG_STRING => value = core::str::from_utf8(body).ok(),
        _ => {}
    });

    if let (Some(name), Some(value)) = (name, value) {
        if !name.is_empty() && !value.is_empty() {
            let mut buf = [0u8; MAX_KEY];
            sink.text(canonical_key(name, album, &mut buf), value);
        }
    }

    // Parent emitted first: a first-value-wins sink then keeps the outer tag when a nested one
    // repeats its name.
    if depth + 1 < MAX_NEST {
        for_each_child(simple, |h, nested| {
            if h.id == id::SIMPLE_TAG {
                emit_simple_tag(nested, album, depth + 1, sink);
            }
        });
    }
}

/// Map a Matroska tag name onto the workspace's Vorbis-shaped key space — see [`parse_tags`]
/// for the table and for what `album` means.
///
/// The fallback is a case fold into `buf`, which the caller owns: a name already upper case
/// (every name in [MatroskaTags]) is returned borrowed, untouched and uncopied.
fn canonical_key<'a>(name: &'a str, album: bool, buf: &'a mut [u8; MAX_KEY]) -> &'a str {
    match name {
        "TITLE" if album => return "ALBUM",
        "ARTIST" if album => return "ALBUMARTIST",
        "PART_NUMBER" => return if album { "DISCNUMBER" } else { "TRACKNUMBER" },
        "TOTAL_PARTS" => return if album { "DISCTOTAL" } else { "TRACKTOTAL" },
        // Matroska splits "when" into what happened when (§ [MatroskaTags]); a listing wants
        // one DATE, and a first-value-wins sink keeps whichever the file states first.
        "DATE_RELEASED" | "DATE_RECORDED" | "DATE_WRITTEN" | "DATE_DIGITIZED" => return "DATE",
        _ => {}
    }
    if name.len() > MAX_KEY || !name.bytes().any(|b| b.is_ascii_lowercase()) {
        return name; // already canonical — the common path, and it borrows
    }
    for (dst, src) in buf.iter_mut().zip(name.bytes()) {
        *dst = src.to_ascii_uppercase();
    }
    // Folding ASCII case cannot change a byte's UTF-8 class, so the result is still valid.
    core::str::from_utf8(&buf[..name.len()]).unwrap_or(name)
}

// =====================================================================================
// 3. Attachments
// =====================================================================================

/// Parse a `\Segment\Attachments` element (RFC 9559 §5.1.7), emitting the image attachments to
/// `sink` as pictures.
///
/// `attachments` is the element **including** its ID + size header, as [`SegmentIndex`] reports
/// it. `FileData` reaches the sink **borrowed** — a 2 MB embedded JPEG is a sub-slice of the
/// caller's read buffer and is never copied here.
///
/// ## What counts as cover art
///
/// Matroska has no dedicated cover-art element; the convention ([MatroskaTags], "Cover Art") is
/// an `AttachedFile` whose `FileName` is `cover.*`, with `small_cover.*` for a thumbnail and
/// the `_land`/`_port` suffixes for orientation variants. Rather than emit only those — which
/// would drop the art from every file that named it `folder.jpg` or `AlbumArt.png` — this emits
/// **every attachment whose `FileMediaType` is an `image/*`**, in two passes so the ones
/// matching the cover convention come first. A sink that keeps its pictures in order therefore
/// finds the real cover at index 0, and a file that attached a fonts-and-subtitles bundle
/// contributes nothing.
pub fn parse_attachments(attachments: &[u8], sink: &mut impl TagSink) {
    let Some(body) = master_body(attachments, id::ATTACHMENTS) else { return };
    // Two passes over the element *structure* only: `FileData` is located, never touched, so
    // the second pass costs a header walk rather than a re-read of the image bytes.
    for cover_pass in [true, false] {
        for_each_child(body, |h, file| {
            if h.id == id::ATTACHED_FILE {
                emit_attachment(file, cover_pass, sink);
            }
        });
    }
}

/// Emit one `AttachedFile` if it is an image and belongs to this pass (see
/// [`parse_attachments`]).
fn emit_attachment(file: &[u8], cover_pass: bool, sink: &mut impl TagSink) {
    let mut name: &str = "";
    let mut mime: Option<&str> = None;
    let mut data: Option<&[u8]> = None;
    for_each_child(file, |h, body| match h.id {
        i if i == id::FILE_NAME => name = core::str::from_utf8(body).unwrap_or(""),
        i if i == id::FILE_MEDIA_TYPE => mime = core::str::from_utf8(body).ok(),
        i if i == id::FILE_DATA => data = Some(body),
        _ => {}
    });

    let (Some(mime), Some(data)) = (mime, data) else { return };
    // The media type is the gate, not the name: an attachment is cover art because it is an
    // image, and `is_cover_name` only decides which pass emits it. Compared case-insensitively
    // *without* folding a copy — `to_ascii_lowercase` would allocate a `String` per attachment,
    // which this path may not do (spec: allocation discipline).
    if strip_prefix_ascii_ci(mime, "image/").is_none() || data.is_empty() {
        return;
    }
    if is_cover_name(name) == cover_pass {
        sink.picture(mime, data);
    }
}

/// Whether a `FileName` follows the cover-art naming convention ([MatroskaTags], "Cover Art"):
/// `cover.*` or `small_cover.*`, optionally with a `_land` / `_port` orientation suffix on the
/// stem. Matched case-insensitively — the convention is lower case, real files are not.
fn is_cover_name(name: &str) -> bool {
    let stem = name.split('.').next().unwrap_or("");
    let stem = strip_suffix_ascii_ci(stem, "_land")
        .or_else(|| strip_suffix_ascii_ci(stem, "_port"))
        .unwrap_or(stem);
    let stem = strip_prefix_ascii_ci(stem, "small_").unwrap_or(stem);
    stem.eq_ignore_ascii_case("cover")
}

/// `str::strip_prefix`, case-insensitively over ASCII — without allocating a folded copy.
fn strip_prefix_ascii_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    let head = s.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix).then(|| &s[prefix.len()..])
}

/// `str::strip_suffix`, case-insensitively over ASCII.
fn strip_suffix_ascii_ci<'a>(s: &'a str, suffix: &str) -> Option<&'a str> {
    let at = s.len().checked_sub(suffix.len())?;
    let tail = s.get(at..)?;
    tail.eq_ignore_ascii_case(suffix).then(|| &s[..at])
}

// =====================================================================================
// 4. Shared walking
// =====================================================================================

/// The body of a master element whose header begins at `data[0]`, checked to be `want`.
///
/// A window holding *more* than the element is clamped to the declared size; one holding less
/// yields what is present, so a truncated read parses as far as it reaches instead of failing.
fn master_body<'a>(data: &'a [u8], want: &[u8]) -> Option<&'a [u8]> {
    let h = ebml::read_element_header(data, 0).ok()?;
    if h.id != want {
        return None;
    }
    // An unknown-size Tags/Attachments master is not something any writer emits, and it has no
    // end to walk to; take the rest of the window.
    let end = match h.data_end() {
        Ok(Some(end)) => end.min(data.len()),
        _ => data.len(),
    };
    data.get(h.data_start..end)
}

/// Call `f` for each direct child of a master's body: its header, and its data as a slice.
///
/// The walk stops — rather than failing the whole parse — at the first child that cannot be
/// read, whose size is unknown, or that runs past the master. That is the total-parse rule:
/// a corrupt element costs the children after the corruption and nothing before it.
fn for_each_child<'a>(body: &'a [u8], mut f: impl FnMut(&ebml::ElementHeader<'a>, &'a [u8])) {
    let mut at = 0usize;
    while at < body.len() {
        let Ok(h) = ebml::read_element_header(body, at) else { return };
        let Ok(Some(end)) = h.data_end() else { return };
        // `end <= at` would be a zero-width element and an infinite loop; a header always has
        // at least the ID and size octets, so this is unreachable for a well-formed one and a
        // hard stop for anything else.
        if end > body.len() || end <= at {
            return;
        }
        f(&h, &body[h.data_start..end]);
        at = end;
    }
}

#[cfg(test)]
// Fixture assembly and the collecting sink are test-only heap use (spec: allocation discipline
// — tests are a sanctioned exception); the parsers under test allocate nothing.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------------
    // Fixtures — built from `ebml`'s own writers, so a test exercises the same byte
    // grammar the muxer emits (the `tests/lacing.rs` pattern).
    // ---------------------------------------------------------------------------------

    /// A master element: ID, the shortest-valid size VINT, then the concatenated children.
    fn master(element_id: &[u8], children: &[Vec<u8>]) -> Vec<u8> {
        let body: Vec<u8> = children.concat();
        let mut out = Vec::new();
        ebml::write_id(&mut out, element_id);
        ebml::write_size(&mut out, body.len() as u64);
        out.extend_from_slice(&body);
        out
    }

    fn uint(element_id: &[u8], v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        ebml::write_uint(&mut out, element_id, v);
        out
    }

    fn utf8(element_id: &[u8], v: &str) -> Vec<u8> {
        let mut out = Vec::new();
        ebml::write_string(&mut out, element_id, v);
        out
    }

    fn binary(element_id: &[u8], v: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        ebml::write_binary(&mut out, element_id, v);
        out
    }

    /// A `SimpleTag` (§5.1.8.1.2) with optional nested children.
    fn simple_tag(name: &str, value: &str, nested: &[Vec<u8>]) -> Vec<u8> {
        let mut kids = vec![utf8(id::TAG_NAME, name), utf8(id::TAG_STRING, value)];
        kids.extend_from_slice(nested);
        master(id::SIMPLE_TAG, &kids)
    }

    /// A `Targets` (§5.1.8.1.1). `level` omitted means the spec default of 50.
    fn targets(level: Option<u64>, track_uid: Option<u64>, chapter_uid: Option<u64>) -> Vec<u8> {
        let mut kids = Vec::new();
        if let Some(l) = level {
            kids.push(uint(id::TARGET_TYPE_VALUE, l));
        }
        if let Some(t) = track_uid {
            kids.push(uint(id::TAG_TRACK_UID, t));
        }
        if let Some(c) = chapter_uid {
            kids.push(uint(id::TAG_CHAPTER_UID, c));
        }
        master(id::TARGETS, &kids)
    }

    /// A `Tag` (§5.1.8.1): its Targets, then its SimpleTags.
    fn tag(targets: Vec<u8>, simples: &[Vec<u8>]) -> Vec<u8> {
        let mut kids = vec![targets];
        kids.extend_from_slice(simples);
        master(id::TAG, &kids)
    }

    fn tags_element(tags: &[Vec<u8>]) -> Vec<u8> {
        master(id::TAGS, tags)
    }

    /// An `AttachedFile` (§5.1.7.1).
    fn attached_file(name: &str, mime: &str, data: &[u8]) -> Vec<u8> {
        master(
            id::ATTACHED_FILE,
            &[
                utf8(id::FILE_NAME, name),
                utf8(id::FILE_MEDIA_TYPE, mime),
                binary(id::FILE_DATA, data),
                uint(id::FILE_UID, 0x1234),
            ],
        )
    }

    /// An `Info` (§5.1.2) declaring a TimestampScale and a Duration in ticks.
    fn info(scale: u64, duration_ticks: f64) -> Vec<u8> {
        let mut dur = Vec::new();
        ebml::write_f64(&mut dur, id::DURATION, duration_ticks);
        master(id::INFO, &[uint(id::TIMESTAMP_SCALE, scale), dur])
    }

    /// A `SeekHead` (§5.1.1) of `(target element ID, Segment Position)` entries.
    fn seek_head(entries: &[(&[u8], u64)]) -> Vec<u8> {
        let seeks: Vec<Vec<u8>> = entries
            .iter()
            .map(|(target, pos)| {
                master(id::SEEK, &[binary(id::SEEK_ID, target), uint(id::SEEK_POSITION, *pos)])
            })
            .collect();
        master(id::SEEK_HEAD, &seeks)
    }

    /// The EBML Header (RFC 8794 §11.2.4) every Matroska file opens with.
    fn ebml_header() -> Vec<u8> {
        master(id::EBML, &[utf8(id::DOC_TYPE, "matroska"), uint(id::DOC_TYPE_VERSION, 4)])
    }

    /// A whole file: the EBML Header, then a definite-size Segment holding `level1`.
    /// Returns the bytes and the absolute offset of the Segment's data.
    fn segment_file(level1: &[Vec<u8>]) -> (Vec<u8>, u64) {
        let header = ebml_header();
        let body: Vec<u8> = level1.concat();
        let mut seg = Vec::new();
        ebml::write_id(&mut seg, id::SEGMENT);
        ebml::write_size(&mut seg, body.len() as u64);
        let data_start = (header.len() + seg.len()) as u64;
        let mut out = header;
        out.extend_from_slice(&seg);
        out.extend_from_slice(&body);
        (out, data_start)
    }

    /// A `Tracks` (§5.1.4) with one audio `TrackEntry`.
    fn tracks(rate: f64, channels: u64) -> Vec<u8> {
        let mut freq = Vec::new();
        ebml::write_f64(&mut freq, id::SAMPLING_FREQUENCY, rate);
        let audio = master(id::AUDIO, &[freq, uint(id::CHANNELS, channels)]);
        let entry = master(
            id::TRACK_ENTRY,
            &[uint(id::TRACK_NUMBER, 1), uint(id::TRACK_TYPE, 2), audio],
        );
        master(id::TRACKS, &[entry])
    }

    /// A `Cluster` of `n` filler bytes — stands in for the megabytes of frames a real file
    /// puts between its front index and its trailing metadata.
    fn cluster(n: usize) -> Vec<u8> {
        master(id::CLUSTER, &[binary(id::SIMPLE_BLOCK, &vec![0u8; n])])
    }

    /// Collects everything a parse emits, in order.
    #[derive(Default)]
    struct Collect {
        text: Vec<(String, String)>,
        pictures: Vec<(String, usize)>,
    }

    impl Collect {
        fn get(&self, key: &str) -> Option<&str> {
            // First value wins, as `ArenaSink`/`TagList` do.
            self.text.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
        }
        fn keys(&self) -> Vec<&str> {
            self.text.iter().map(|(k, _)| k.as_str()).collect()
        }
    }

    impl TagSink for Collect {
        fn text(&mut self, key: &str, value: &str) {
            self.text.push((key.to_string(), value.to_string()));
        }
        fn picture(&mut self, mime: &str, data: &[u8]) {
            self.pictures.push((mime.to_string(), data.len()));
        }
    }

    fn parse(tags: &[u8]) -> Collect {
        let mut c = Collect::default();
        parse_tags(tags, &mut c);
        c
    }

    // ---------------------------------------------------------------------------------
    // Tags
    // ---------------------------------------------------------------------------------

    /// The plain case: one untargeted `Tag` whose `SimpleTag`s carry the usual vocabulary.
    /// Matroska's own spellings map onto the workspace's Vorbis-shaped keys.
    #[test]
    fn a_flat_tag_block_maps_onto_canonical_keys() {
        let t = tags_element(&[tag(
            targets(None, None, None),
            &[
                simple_tag("TITLE", "Enough", &[]),
                simple_tag("ARTIST", "Fred again..", &[]),
                simple_tag("ALBUM", "Actual Life 3", &[]),
                simple_tag("DATE_RELEASED", "2022-10-28", &[]),
                simple_tag("PART_NUMBER", "7", &[]),
                simple_tag("TOTAL_PARTS", "12", &[]),
                simple_tag("GENRE", "Électronique", &[]),
            ],
        )]);
        let c = parse(&t);
        assert_eq!(c.get("TITLE"), Some("Enough"));
        assert_eq!(c.get("ARTIST"), Some("Fred again.."));
        assert_eq!(c.get("ALBUM"), Some("Actual Life 3"));
        assert_eq!(c.get("DATE"), Some("2022-10-28"), "DATE_RELEASED is the file's DATE");
        assert_eq!(c.get("TRACKNUMBER"), Some("7"), "PART_NUMBER at track scope");
        assert_eq!(c.get("TRACKTOTAL"), Some("12"));
        assert_eq!(c.get("GENRE"), Some("Électronique"), "non-ASCII UTF-8 passes through");
    }

    /// A name Matroska does not define passes through, folded to upper case; one already
    /// upper case is borrowed untouched.
    #[test]
    fn unknown_names_pass_through_uppercased() {
        let t = tags_element(&[tag(
            targets(None, None, None),
            &[
                simple_tag("ENCODER", "Lavf60.3.100", &[]),
                simple_tag("mood", "restless", &[]),
                simple_tag("Replaygain_Track_Gain", "-7.35 dB", &[]),
            ],
        )]);
        let c = parse(&t);
        assert_eq!(c.get("ENCODER"), Some("Lavf60.3.100"));
        assert_eq!(c.get("MOOD"), Some("restless"));
        assert_eq!(c.get("REPLAYGAIN_TRACK_GAIN"), Some("-7.35 dB"));
    }

    /// A name longer than the fold buffer is passed through as written rather than truncated
    /// — a wrong key is worse than an unfolded one.
    #[test]
    fn an_overlong_name_is_not_truncated() {
        let long = "x".repeat(MAX_KEY + 10);
        let t = tags_element(&[tag(targets(None, None, None), &[simple_tag(&long, "v", &[])])]);
        let c = parse(&t);
        assert_eq!(c.keys(), vec![long.as_str()], "passed through unfolded, not cut short");
    }

    /// With **no** track-level `Tag`, a lone default-level (50) block is the recording's own
    /// metadata — the shape almost every real file has, and the one where remapping `TITLE`
    /// to `ALBUM` would leave the file with no title.
    #[test]
    fn a_lone_default_level_block_is_the_tracks_own() {
        let t = tags_element(&[tag(
            targets(None, None, None),
            &[simple_tag("TITLE", "Song", &[]), simple_tag("ARTIST", "Band", &[])],
        )]);
        let c = parse(&t);
        assert_eq!(c.get("TITLE"), Some("Song"));
        assert_eq!(c.get("ARTIST"), Some("Band"));
        assert_eq!(c.get("ALBUM"), None);
        assert_eq!(c.get("ALBUMARTIST"), None);
    }

    /// …but when the file *does* distinguish the levels, the album block's TITLE/ARTIST are
    /// the album's, and only the track block's are the track's (§5.1.8.1.1 Table 33).
    #[test]
    fn album_level_remaps_when_a_track_level_exists() {
        let t = tags_element(&[
            tag(
                targets(Some(50), None, None),
                &[
                    simple_tag("TITLE", "Actual Life 3", &[]),
                    simple_tag("ARTIST", "Fred again..", &[]),
                    simple_tag("PART_NUMBER", "2", &[]),
                    simple_tag("TOTAL_PARTS", "3", &[]),
                ],
            ),
            tag(
                targets(Some(30), None, None),
                &[
                    simple_tag("TITLE", "Enough", &[]),
                    simple_tag("ARTIST", "Fred again..", &[]),
                    simple_tag("PART_NUMBER", "7", &[]),
                ],
            ),
        ]);
        let c = parse(&t);
        assert_eq!(c.get("ALBUM"), Some("Actual Life 3"));
        assert_eq!(c.get("ALBUMARTIST"), Some("Fred again.."));
        assert_eq!(c.get("DISCNUMBER"), Some("2"), "PART_NUMBER at album level is the disc");
        assert_eq!(c.get("DISCTOTAL"), Some("3"));
        assert_eq!(c.get("TITLE"), Some("Enough"), "the track block keeps TITLE");
        assert_eq!(c.get("ARTIST"), Some("Fred again.."));
        assert_eq!(c.get("TRACKNUMBER"), Some("7"));
    }

    /// Tags describing a *part* of the file are dropped: SUBTRACK (20) and SHOT (10) are below
    /// the track level, and a non-zero `TagChapterUID` pins a tag to one chapter.
    #[test]
    fn sub_track_and_chapter_scoped_tags_are_skipped() {
        let t = tags_element(&[
            tag(targets(Some(30), None, None), &[simple_tag("TITLE", "The recording", &[])]),
            tag(targets(Some(20), None, None), &[simple_tag("TITLE", "Movement II", &[])]),
            tag(targets(Some(10), None, None), &[simple_tag("TITLE", "A shot", &[])]),
            tag(targets(Some(30), None, Some(99)), &[simple_tag("TITLE", "Chapter 4", &[])]),
        ]);
        let c = parse(&t);
        assert_eq!(c.get("TITLE"), Some("The recording"));
        assert_eq!(c.keys().len(), 1, "only the unscoped track-level tag survives");
    }

    /// A `TagTrackUID` is *not* a disqualifier — it names which track a tag applies to, and a
    /// one-file scan wants it. (The real ffmpeg-muxed webm files in the corpus are shaped
    /// exactly like this: an untargeted ENCODER plus a per-track DURATION.)
    #[test]
    fn a_track_scoped_tag_is_kept() {
        let t = tags_element(&[
            tag(targets(None, None, None), &[simple_tag("ENCODER", "Lavf60.3.100", &[])]),
            tag(
                targets(None, Some(0x9A7B_C3D2), None),
                &[simple_tag("DURATION", "00:59:37.640000000", &[])],
            ),
        ]);
        let c = parse(&t);
        assert_eq!(c.get("ENCODER"), Some("Lavf60.3.100"));
        assert_eq!(c.get("DURATION"), Some("00:59:37.640000000"));
    }

    /// Nested `SimpleTag`s are flattened, parent first so a first-value-wins sink keeps the
    /// outer tag when a child repeats its name.
    #[test]
    fn nested_simple_tags_are_flattened_parent_first() {
        let t = tags_element(&[tag(
            targets(None, None, None),
            &[simple_tag(
                "ARTIST",
                "The Beatles",
                &[
                    simple_tag("SORT_WITH", "Beatles, The", &[]),
                    simple_tag("INSTRUMENTS", "guitar", &[]),
                ],
            )],
        )]);
        let c = parse(&t);
        assert_eq!(c.keys(), vec!["ARTIST", "SORT_WITH", "INSTRUMENTS"], "parent precedes children");
        assert_eq!(c.get("ARTIST"), Some("The Beatles"));
        assert_eq!(c.get("SORT_WITH"), Some("Beatles, The"));
    }

    /// Nesting is followed to [`MAX_NEST`] and no further, so a crafted chain is a bounded
    /// walk rather than a blown stack.
    #[test]
    fn nesting_is_depth_capped() {
        // Build MAX_NEST + 8 levels, innermost first.
        let mut inner = simple_tag("LEVEL", "deepest", &[]);
        for i in 0..MAX_NEST + 8 {
            inner = simple_tag("LEVEL", &format!("{i}"), &[inner]);
        }
        let t = tags_element(&[tag(targets(None, None, None), &[inner])]);
        let c = parse(&t);
        assert_eq!(c.keys().len(), MAX_NEST, "exactly the cap, and no recursion past it");
    }

    /// A `SimpleTag` missing either half of the pair, carrying an empty one, or holding bytes
    /// that are not UTF-8, emits nothing rather than a placeholder.
    #[test]
    fn incomplete_or_non_utf8_pairs_emit_nothing() {
        let name_only = master(id::SIMPLE_TAG, &[utf8(id::TAG_NAME, "TITLE")]);
        let value_only = master(id::SIMPLE_TAG, &[utf8(id::TAG_STRING, "orphan")]);
        let empty_value = simple_tag("TITLE", "", &[]);
        // 0xFF is never a valid UTF-8 byte (RFC 3629).
        let bad_utf8 = master(
            id::SIMPLE_TAG,
            &[utf8(id::TAG_NAME, "TITLE"), binary(id::TAG_STRING, &[0xFF, 0xFE])],
        );
        let t = tags_element(&[tag(
            targets(None, None, None),
            &[name_only, value_only, empty_value, bad_utf8, simple_tag("ARTIST", "real", &[])],
        )]);
        let c = parse(&t);
        assert_eq!(c.keys(), vec!["ARTIST"], "only the well-formed pair is emitted");
    }

    /// A `Tags` element with no `Tag`, a `Tag` with no `SimpleTag`, and a window that is not a
    /// `Tags` element at all: all emit nothing, none panic.
    #[test]
    fn degenerate_tags_elements_emit_nothing() {
        assert!(parse(&tags_element(&[])).text.is_empty());
        assert!(parse(&tags_element(&[tag(targets(None, None, None), &[])])).text.is_empty());
        assert!(parse(&info(1_000_000, 5.0)).text.is_empty(), "wrong element ID");
        assert!(parse(&[]).text.is_empty());
        assert!(parse(&[0xFF, 0xFF, 0xFF]).text.is_empty());
    }

    // ---------------------------------------------------------------------------------
    // Attachments
    // ---------------------------------------------------------------------------------

    fn attach(files: &[Vec<u8>]) -> Collect {
        let mut c = Collect::default();
        parse_attachments(&master(id::ATTACHMENTS, files), &mut c);
        c
    }

    /// Image attachments become pictures; the one following the `cover.*` convention comes
    /// first whatever order the file wrote them in, and non-images are ignored entirely.
    #[test]
    fn attachments_emit_images_cover_first() {
        let c = attach(&[
            attached_file("banner.png", "image/png", &[1u8; 40]),
            attached_file("subtitles.srt", "text/plain", &[2u8; 10]),
            attached_file("cover.jpg", "image/jpeg", &[3u8; 100]),
            attached_file("DejaVuSans.ttf", "application/x-truetype-font", &[4u8; 20]),
        ]);
        assert_eq!(
            c.pictures,
            vec![("image/jpeg".to_string(), 100), ("image/png".to_string(), 40)],
            "cover.jpg first, then the other image; the font and the subtitles are not pictures"
        );
    }

    /// The cover convention's variants ([MatroskaTags]): `small_cover`, the `_land`/`_port`
    /// orientation suffixes, and any casing.
    #[test]
    fn the_cover_naming_convention_is_matched_loosely() {
        for name in ["cover.jpg", "Cover.PNG", "small_cover.jpg", "cover_land.jpg", "COVER_PORT.png"]
        {
            assert!(is_cover_name(name), "{name} is a cover");
        }
        for name in ["banner.png", "coverart.jpg", "folder.jpg", "", "recover.png"] {
            assert!(!is_cover_name(name), "{name} is not");
        }
    }

    /// An attachment missing its media type or its data, or declaring a non-image type, is
    /// not a picture. An `image/*` type with unusual casing still is.
    #[test]
    fn only_image_attachments_with_data_become_pictures() {
        let no_mime = master(id::ATTACHED_FILE, &[utf8(id::FILE_NAME, "cover.jpg")]);
        let no_data = master(
            id::ATTACHED_FILE,
            &[utf8(id::FILE_NAME, "cover.jpg"), utf8(id::FILE_MEDIA_TYPE, "image/jpeg")],
        );
        let empty = attached_file("cover.jpg", "image/jpeg", &[]);
        let c = attach(&[no_mime, no_data, empty, attached_file("a.webp", "IMAGE/WebP", &[9u8; 7])]);
        assert_eq!(c.pictures, vec![("IMAGE/WebP".to_string(), 7)]);
    }

    // ---------------------------------------------------------------------------------
    // Tracks
    // ---------------------------------------------------------------------------------

    /// The first audio track's rate and channel count, off the `Audio` master (§5.1.4.1.28).
    #[test]
    fn audio_format_reads_the_first_audio_track() {
        assert_eq!(audio_format(&tracks(48_000.0, 2)), Some((48_000, 2)));

        // A video track first: the walk keeps looking rather than reporting its (absent) audio.
        let video = master(
            id::TRACK_ENTRY,
            &[uint(id::TRACK_NUMBER, 1), uint(id::TRACK_TYPE, 1)],
        );
        let mut freq = Vec::new();
        ebml::write_f32(&mut freq, id::SAMPLING_FREQUENCY, 44_100.0);
        let audio_entry = master(
            id::TRACK_ENTRY,
            &[
                uint(id::TRACK_NUMBER, 2),
                uint(id::TRACK_TYPE, 2),
                master(id::AUDIO, &[freq, uint(id::CHANNELS, 6)]),
            ],
        );
        assert_eq!(
            audio_format(&master(id::TRACKS, &[video, audio_entry])),
            Some((44_100, 6)),
            "a 4-octet float reads too, and the video track is stepped over"
        );
    }

    /// A file with no audio track, a malformed rate, and a window that is not a `Tracks`
    /// element: nothing stated, never a panic.
    #[test]
    fn audio_format_declines_what_it_cannot_read() {
        let video_only = master(
            id::TRACKS,
            &[master(id::TRACK_ENTRY, &[uint(id::TRACK_TYPE, 1)])],
        );
        assert_eq!(audio_format(&video_only), None);
        assert_eq!(audio_format(&master(id::TRACKS, &[])), None);
        assert_eq!(audio_format(&tags_element(&[])), None, "wrong element ID");
        assert_eq!(audio_format(&[]), None);

        // A negative, non-finite, or wrong-width SamplingFrequency reports 0 — "not stated" —
        // rather than a nonsense rate, and the channel count still comes through.
        for bad in [-48_000.0f64, f64::NAN, f64::INFINITY] {
            assert_eq!(audio_format(&tracks(bad, 2)), Some((0, 2)), "{bad} is not a rate");
        }
        let odd_width = master(
            id::TRACKS,
            &[master(
                id::TRACK_ENTRY,
                &[
                    uint(id::TRACK_TYPE, 2),
                    master(id::AUDIO, &[binary(id::SAMPLING_FREQUENCY, &[1, 2, 3])]),
                ],
            )],
        );
        assert_eq!(audio_format(&odd_width), Some((0, 0)));
    }

    // ---------------------------------------------------------------------------------
    // Locating and measuring
    // ---------------------------------------------------------------------------------

    /// The layout the corpus actually has: everything indexed by a front SeekHead, with the
    /// masters themselves also present in the window. The elements win; every reported offset
    /// is the first byte of the named element's header.
    #[test]
    fn index_segment_finds_the_masters_in_the_window() {
        let level1 = [
            seek_head(&[(id::INFO, 0), (id::TAGS, 0)]), // positions deliberately wrong
            info(1_000_000, 3_000.0),
            tracks(48_000.0, 2),
            tags_element(&[tag(targets(None, None, None), &[simple_tag("TITLE", "T", &[])])]),
            cluster(64),
        ];
        let (file, data_start) = segment_file(&level1);
        let ix = index_segment(&file).expect("a Segment");
        assert_eq!(ix.segment_data_start, data_start);

        let info_at = ix.info.expect("Info") as usize;
        let tags_at = ix.tags.expect("Tags") as usize;
        assert_eq!(&file[info_at..info_at + id::INFO.len()], id::INFO, "points at the Info header");
        assert_eq!(&file[tags_at..tags_at + id::TAGS.len()], id::TAGS);
        assert_eq!(ix.attachments, None, "the file has none");
        let tracks_at = ix.tracks.expect("Tracks") as usize;
        assert_eq!(audio_format(&file[tracks_at..]), Some((48_000, 2)));
        assert_eq!(info_duration_ns(&file[info_at..]), Some(3_000_000_000));
        assert_eq!(parse(&file[tags_at..]).get("TITLE"), Some("T"));
    }

    /// The layout that makes the SeekHead worth having: `Tags` and `Attachments` written
    /// *after* the clusters, past a prefix a scanner is willing to read. The offsets come out
    /// of the index alone, and reading there finds the elements.
    #[test]
    fn index_segment_reaches_trailing_masters_through_the_seek_head() {
        let filler = 200_000;
        let tags = tags_element(&[tag(targets(None, None, None), &[simple_tag("TITLE", "Far", &[])])]);
        let attachments =
            master(id::ATTACHMENTS, &[attached_file("cover.jpg", "image/jpeg", &[7u8; 5_000])]);

        // The SeekHead's own length depends on the positions it carries, which depend on its
        // length — so settle the two by iteration, which is the same fixed point a muxer's
        // finalize reaches by reserving the space first. The VINT width only grows, so this
        // converges in a couple of rounds.
        let (mut tags_pos, mut att_pos) = (0u64, 0u64);
        for _ in 0..8 {
            let head = [
                seek_head(&[(id::TAGS, tags_pos), (id::ATTACHMENTS, att_pos)]),
                info(1_000_000, 1.0),
                cluster(filler),
            ]
            .concat();
            tags_pos = head.len() as u64;
            att_pos = tags_pos + tags.len() as u64;
        }

        let level1 = [
            seek_head(&[(id::TAGS, tags_pos), (id::ATTACHMENTS, att_pos)]),
            info(1_000_000, 1.0),
            cluster(filler),
            tags.clone(),
            attachments.clone(),
        ];
        let (file, data_start) = segment_file(&level1);

        // A scanner holds only the prefix — far short of the trailing metadata.
        let prefix = &file[..64 * 1024];
        let ix = index_segment(prefix).expect("a Segment");
        assert_eq!(ix.segment_data_start, data_start);
        let tags_at = ix.tags.expect("Tags via the SeekHead") as usize;
        let att_at = ix.attachments.expect("Attachments via the SeekHead") as usize;
        assert!(tags_at > prefix.len(), "the tags really are past the prefix");
        assert_eq!(&file[tags_at..tags_at + tags.len()], &tags[..], "the position is exact");
        assert_eq!(&file[att_at..att_at + attachments.len()], &attachments[..]);

        // …and the caller's read-then-ask-again loop terminates in one hop.
        assert_eq!(element_extent(&file[tags_at..tags_at + 8]), Extent::Short {
            total: tags.len() as u64
        });
        assert_eq!(element_extent(&file[tags_at..]), Extent::Complete { len: tags.len() });
        assert_eq!(parse(&file[tags_at..tags_at + tags.len()]).get("TITLE"), Some("Far"));

        let mut c = Collect::default();
        parse_attachments(&file[att_at..att_at + attachments.len()], &mut c);
        assert_eq!(c.pictures, vec![("image/jpeg".to_string(), 5_000)]);
    }

    /// `Duration` is a float in TimestampScale ticks (§5.1.2). Both the default scale and a
    /// declared one convert, and a file that declares no duration says so.
    #[test]
    fn info_duration_applies_the_timestamp_scale() {
        // 3000 ticks at the default 1 ms/tick.
        assert_eq!(info_duration_ns(&info(1_000_000, 3_000.0)), Some(3_000_000_000));
        // The same 3 s at a 100 µs scale is ten times the ticks.
        assert_eq!(info_duration_ns(&info(100_000, 30_000.0)), Some(3_000_000_000));
        // A fractional tick count rounds rather than truncating.
        assert_eq!(info_duration_ns(&info(1_000_000, 1_234.5)), Some(1_234_500_000));
        // No Duration child — a live or still-being-written stream.
        assert_eq!(info_duration_ns(&master(id::INFO, &[uint(id::TIMESTAMP_SCALE, 1_000_000)])), None);
        // Not an Info element, and junk.
        assert_eq!(info_duration_ns(&tags_element(&[])), None);
        assert_eq!(info_duration_ns(&[]), None);
    }

    /// A corrupt duration is not a length: negative, infinite and NaN all report nothing.
    #[test]
    fn a_corrupt_duration_is_rejected() {
        for bad in [-1.0f64, f64::INFINITY, f64::NAN, f64::MAX] {
            assert_eq!(info_duration_ns(&info(1_000_000, bad)), None, "{bad} is not a duration");
        }
    }

    /// [`element_extent`] answers the three cases the read loop needs, and refuses an
    /// unknown-size master (§6.2), which has no length to read.
    #[test]
    fn element_extent_measures_or_asks_for_more() {
        let t = tags_element(&[tag(targets(None, None, None), &[simple_tag("A", "b", &[])])]);
        assert_eq!(element_extent(&t), Extent::Complete { len: t.len() });
        // A window holding more than the element still measures the element.
        let mut padded = t.clone();
        padded.extend_from_slice(&[0u8; 100]);
        assert_eq!(element_extent(&padded), Extent::Complete { len: t.len() });
        // A window holding the header but not all the data names the total (the Tags header is
        // a 4-octet ID plus a 1-octet size).
        assert_eq!(element_extent(&t[..6]), Extent::Short { total: t.len() as u64 });
        assert_eq!(element_extent(&[]), Extent::Invalid);
        assert_eq!(element_extent(&[0x00, 0x01]), Extent::Invalid, "a VINT wider than 8 octets");

        let mut unknown = Vec::new();
        ebml::write_id(&mut unknown, id::SEGMENT);
        ebml::write_unknown_size(&mut unknown);
        assert_eq!(element_extent(&unknown), Extent::Invalid, "no declared end to read to");
    }

    /// Not a Matroska file, or one truncated before the Segment header: no index, no panic.
    #[test]
    fn index_segment_declines_what_is_not_a_segment() {
        assert_eq!(index_segment(&[]), None);
        assert_eq!(index_segment(b"ID3\x04\x00\x00\x00\x00\x00\x00"), None);
        let (file, _) = segment_file(&[info(1_000_000, 1.0)]);
        // Truncated inside the EBML Header, before the Segment is reached.
        assert_eq!(index_segment(&file[..6]), None);
    }

    // ---------------------------------------------------------------------------------
    // Untrusted input
    // ---------------------------------------------------------------------------------

    /// A SeekHead entry pointing at itself, or at an absurd position, is followed at most
    /// once and never loops (§5.1.1 permits SeekHead chains).
    #[test]
    fn a_self_referential_seek_head_terminates() {
        let level1 = [seek_head(&[(id::SEEK_HEAD, 0), (id::TAGS, u64::MAX)]), info(1_000_000, 1.0)];
        let (file, _) = segment_file(&level1);
        let ix = index_segment(&file).expect("a Segment");
        // The Tags position saturates rather than wrapping, and the Info in the window is found.
        assert_eq!(ix.tags, Some(u64::MAX));
        assert!(ix.info.is_some());
    }

    /// Every truncation of a full fixture — locate, measure, and both parsers — runs without
    /// panicking (spec: "a crash on bad input is a P0").
    #[test]
    fn truncation_never_panics() {
        let level1 = [
            seek_head(&[(id::INFO, 0), (id::TAGS, 40), (id::ATTACHMENTS, 90)]),
            info(1_000_000, 4_242.0),
            tags_element(&[
                tag(
                    targets(Some(50), Some(7), None),
                    &[simple_tag("TITLE", "Album", &[simple_tag("SORT_WITH", "s", &[])])],
                ),
                tag(targets(Some(30), None, None), &[simple_tag("PART_NUMBER", "3", &[])]),
            ]),
            master(id::ATTACHMENTS, &[attached_file("cover.jpg", "image/jpeg", &[0xAB; 300])]),
            cluster(32),
        ];
        let (file, _) = segment_file(&level1);
        for n in 0..file.len() {
            let w = &file[..n];
            let _ = index_segment(w);
            let _ = element_extent(w);
            let _ = info_duration_ns(w);
            let mut c = Collect::default();
            parse_tags(w, &mut c);
            parse_attachments(w, &mut c);
        }
    }

    /// Bytes that are structurally hostile rather than merely short: sizes that overflow, a
    /// zero-width element, and a child claiming to run past its master.
    #[test]
    fn hostile_structures_never_panic() {
        let mut c = Collect::default();
        // A Tags master declaring 2^56-2 octets of children in a 12-byte buffer.
        let mut huge = Vec::new();
        ebml::write_id(&mut huge, id::TAGS);
        ebml::write_size(&mut huge, ebml::MAX_VINT_DATA);
        huge.extend_from_slice(&[0u8; 4]);
        parse_tags(&huge, &mut c);
        parse_attachments(&huge, &mut c);

        // A Tag whose declared size runs past the Tags master that contains it.
        let mut inner = Vec::new();
        ebml::write_id(&mut inner, id::TAG);
        ebml::write_size(&mut inner, 10_000);
        let mut outer = Vec::new();
        ebml::write_id(&mut outer, id::TAGS);
        ebml::write_size(&mut outer, inner.len() as u64);
        outer.extend_from_slice(&inner);
        parse_tags(&outer, &mut c);

        // Every single-byte and two-byte window: none is a valid element, none may panic.
        for a in 0u8..=255 {
            parse_tags(&[a], &mut c);
            parse_attachments(&[a], &mut c);
            let _ = index_segment(&[a]);
            let _ = element_extent(&[a]);
            let _ = info_duration_ns(&[a]);
            for b in [0u8, 1, 0x7F, 0x80, 0xFF] {
                parse_tags(&[a, b], &mut c);
                let _ = element_extent(&[a, b]);
            }
        }
        assert!(c.text.is_empty() && c.pictures.is_empty(), "nothing valid was in any of that");
    }
}
