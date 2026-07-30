//! In-band events: travel *with* buffers through the same queues, ordered relative
//! to them (spec: Events, queries, and the bus). Delivered to `Element::event()`
//! at batch boundaries. There are no "non-serialized" events — the out-of-band
//! cases (flush/seek, a live property set) are pipeline operations the *scheduler*
//! delivers through the same `event()` door at the same batch-boundary safe point.

use crate::format::{FixedFormat, Value};
use crate::memory::Memory;
use crate::time::Timestamp;

pub enum Event {
    Segment { base: Timestamp, rate: f64 },
    FormatChange(FixedFormat),
    /// "Nothing through running time T" — the sparse-stream heartbeat, so a silent
    /// subtitle/audio track never stalls preroll or aggregation (spec: GAP protocol).
    Gap { until: Timestamp },
    Tags(TagList),
    /// A live property changed (spec: Dynamic element properties). Scheduler-delivered
    /// at the batch boundary — never mid-buffer; a set issued while batch N is in
    /// flight takes effect no earlier than N+1. `name` is the entry from this
    /// element's own `desc().props`; the value has already been validated against
    /// that entry's constraint. Elements that re-read via
    /// [`Ctx::prop`](crate::ctx::Ctx::prop) each batch may ignore this event.
    PropChanged { name: &'static str, value: Value },
    Eos,
    FlushStart,
    FlushStop,
    /// The transport paused (spec: Clocking — "pause is a clock op, not a state"):
    /// running time freezes, the scheduler parks the group after delivering this.
    /// A device sink reacts by holding its hardware (render silence, keep the ring);
    /// most elements ignore it.
    Paused,
    /// The transport resumed: running time continues (the pause interval is excised
    /// by re-basing), delivered just before the group runs again.
    Resumed,
    /// A positioned overwrite of bytes the sink has already written: `data` replaces
    /// the bytes at absolute output offset `offset`. The **single back-patch** an
    /// indexed container needs (a Matroska SeekHead reservation, an MP4 `moov` size)
    /// while the stream itself stays single-pass. In-band like every event — it rides
    /// after the bytes it patches, so a seekable byte sink applies it as one
    /// positioned write; anything non-seekable (a socket) ignores it, which is
    /// correct: indexed layouts only serve seekable targets. Note a pure transport
    /// (the queue) consumes inbound events and must forward this one explicitly,
    /// like `FormatChange`.
    Patch { offset: u64, data: Memory },
}

/// Stream/container metadata carried in-band by [`Event::Tags`] and mirrored to the application as
/// [`BusMessage::Tags`](crate::bus::BusMessage::Tags) — profluens' analogue of a GStreamer
/// `GstTagList` (a sticky tag event travelling downstream *and* a tag message on the bus). A
/// demuxer/parser fills one from its header (Vorbis comments, an ID3/iTunes atom, a Matroska
/// `Tags` element) and emits it; a muxer consumes it to write its own container's tag encoding; a
/// player reads it off the bus for "now playing" / cover art.
///
/// Text tags use the Vorbis-comment field-name vocabulary, uppercased (`TITLE`, `ARTIST`, `ALBUM`,
/// `DATE`, `TRACKNUMBER`, …), and a key may repeat (two `ARTIST`s — kept in order). Pictures (cover
/// art) hold their bytes zero-copy in [`Memory`], so forwarding art through the graph never copies
/// it.
#[derive(Clone, Default)]
pub struct TagList {
    text: Vec<(Box<str>, Box<str>)>,
    pictures: Vec<Picture>,
}

/// An attached picture (cover art): its MIME type and the image bytes, held zero-copy.
#[derive(Clone)]
pub struct Picture {
    pub mime: Box<str>,
    pub data: Memory,
}

impl TagList {
    /// An empty tag list.
    pub fn new() -> Self {
        Self::default()
    }

    /// No text tags and no pictures.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.pictures.is_empty()
    }

    /// Number of text tags plus pictures.
    pub fn len(&self) -> usize {
        self.text.len() + self.pictures.len()
    }

    /// Append a text tag; `key` is normalised to its uppercase Vorbis-comment field name.
    // One-time: a demuxer/parser builds a `TagList` once per stream (spec: allocation discipline —
    // header parsing is the sanctioned one-time-allocation exception), never in the per-frame path.
    #[allow(clippy::disallowed_methods)]
    pub fn add(&mut self, key: &str, value: &str) -> &mut Self {
        self.text.push((key.to_ascii_uppercase().into(), value.into()));
        self
    }

    /// Append an attached picture (cover art), keeping the image bytes zero-copy.
    #[allow(clippy::disallowed_methods)] // one-time — see `add`
    pub fn add_picture(&mut self, mime: &str, data: Memory) -> &mut Self {
        self.pictures.push(Picture { mime: mime.into(), data });
        self
    }

    /// First value for `key` (case-insensitive), if present.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.text.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)).map(|(_, v)| &**v)
    }

    /// All `(key, value)` text tags, in insertion order (a key may repeat).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.text.iter().map(|(k, v)| (&**k, &**v))
    }

    /// The attached pictures (cover art).
    pub fn pictures(&self) -> &[Picture] {
        &self.pictures
    }

    /// Fold `other`'s tags into `self` — a later parsing stage refining an earlier stage's tags.
    pub fn merge(&mut self, other: TagList) {
        self.text.extend(other.text);
        self.pictures.extend(other.pictures);
    }
}

#[cfg(test)]
mod tag_tests {
    use super::TagList;

    #[test]
    fn text_tags_add_get_iter() {
        let mut t = TagList::new();
        assert!(t.is_empty());
        t.add("title", "Enough")
            .add("Artist", "Fred again..")
            .add("ARTIST", "Brian Eno");

        // Keys are uppercased and looked up case-insensitively; `get` returns the first value.
        assert_eq!(t.get("TITLE"), Some("Enough"));
        assert_eq!(t.get("title"), Some("Enough"));
        assert_eq!(t.get("artist"), Some("Fred again.."));
        assert_eq!(t.get("album"), None);

        // Repeated keys are kept in order.
        let all: Vec<_> = t.iter().collect();
        assert_eq!(
            all,
            vec![("TITLE", "Enough"), ("ARTIST", "Fred again.."), ("ARTIST", "Brian Eno")]
        );
        assert_eq!(t.len(), 3);
        assert!(!t.is_empty());
    }

    #[test]
    fn merge_appends() {
        let mut a = TagList::new();
        a.add("title", "A");
        let mut b = TagList::new();
        b.add("genre", "Ambient");
        a.merge(b);
        assert_eq!(a.get("TITLE"), Some("A"));
        assert_eq!(a.get("GENRE"), Some("Ambient"));
    }
}
