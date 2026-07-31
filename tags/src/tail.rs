//! The trailing tag pair — APEv2 then ID3v1 — shared by every format that keeps metadata at
//! the end of the file.
//!
//! Four of this scanner's formats have no container field for a title, so a tagger bolts one
//! onto the end. The layout is the same in all four:
//!
//! ```text
//!   | … stream … | APEv2 tag | ID3v1 (128 B) | EOF
//! ```
//!
//! ID3v1 is by definition the last 128 bytes (Eric Kemp's 1996 `id3v1` note), and an APEv2
//! tag sits *in front of* it — `APEv2_specification`, "Tag location": a tag written by a
//! ReplayGain tool must not disturb a trailing v1 block, so the APE footer has to be looked
//! for at `file_len - 128` as well as at `file_len`. Who says so, per format:
//!
//! | format | authority |
//! |---|---|
//! | MP3 | de-facto; the elementary stream has nowhere else to put a tag (`src/mp3.rs`) |
//! | Monkey's Audio | `spec/APE.md`, "Tags" — APEv2's own reference format |
//! | WavPack | `spec/WavPack5FileFormat.txt` §4.0: "Both the APEv2 tags and/or ID3v1 tags must come at the end of the WavPack file, with the ID3v1 coming last if both are present." |
//! | Musepack | `spec/MPC.md`, "Tags" |
//!
//! One implementation, measured once, tested once. The parsing itself is [`pf_mp3::id3`]'s:
//! that crate owns APEv2 and ID3v1 because MP3 needed them first, and nothing about either
//! block is MP3-specific.
//!
//! ## What it costs
//!
//! **One read.** [`measure`] and [`emit`] both work on a window whose last byte is the last
//! byte of the file — the engine's single `tail_len` read, which is skipped outright when the
//! prefix already holds the whole file. Nothing here allocates: the blocks are parsed where
//! they were read and values reach the sink borrowed from them.

use profluens_core::event::TagSink;
use profluens_core::memory::Arena;

/// An ID3v1 tag is a fixed 128-byte block at the very end of the file (Eric Kemp's 1996
/// `id3v1` note).
pub(crate) const V1_LEN: u64 = 128;

/// An APEv2 header and footer share one 32-octet layout (`APEv2_specification`); the footer
/// is the minimum [`pf_mp3::id3::ape_tail_len`] can be asked about.
const APE_FOOTER: u64 = 32;

/// The trailing side-cars, in bytes.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct Trailers {
    /// 128 when the file ends with an ID3v1 tag, else 0.
    pub v1: u64,
    /// Length of the APEv2 block in front of it (header included), else 0.
    pub ape: u64,
}

impl Trailers {
    /// Bytes at the end of the file that are metadata, not stream — what a size-derived
    /// duration estimate must not count.
    pub(crate) fn total(self) -> u64 {
        self.v1 + self.ape
    }
}

/// Measure the trailers in `tail` — a window whose first byte is at file offset `base` and
/// whose last byte is the last byte of the file. A window that does not end at EOF measures
/// nothing rather than guessing: every offset below is derived from `file_len`.
pub(crate) fn measure(tail: &[u8], base: u64, file_len: u64) -> Trailers {
    let mut t = Trailers::default();
    if file_len.checked_sub(base) != Some(tail.len() as u64) {
        return t;
    }
    // Relative position of an absolute file offset, `None` if it is outside the window.
    let rel = |abs: u64| -> Option<usize> {
        usize::try_from(abs.checked_sub(base)?).ok().filter(|&d| d <= tail.len())
    };

    // ID3v1: the last 128 bytes, identified by its "TAG" magic.
    if let Some(at) = file_len.checked_sub(V1_LEN).and_then(&rel) {
        if tail[at..].starts_with(b"TAG") {
            t.v1 = V1_LEN;
        }
    }
    // APEv2: its footer occupies the last 32 bytes of whatever precedes ID3v1.
    let ape_end = file_len - t.v1;
    if let Some(at) = rel(ape_end).filter(|&at| at as u64 >= APE_FOOTER) {
        // `ape_tail_len` reads the *last* 32 bytes of the slice it is given, so the slice
        // ends exactly where the APE tag would.
        if let Some(len) = pf_mp3::id3::ape_tail_len(&tail[..at]).filter(|&n| n as u64 <= ape_end) {
            t.ape = len as u64;
        }
    }
    t
}

/// Emit the trailing tags, **after** whatever the head of the file contributed has gone into
/// the same sink.
///
/// Order is APEv2 then ID3v1, which is pf-mp3's documented precedence: a sink's `get` returns
/// the first value for a key, so a richer head tag (ID3v2, or an APE/WavPack/MPC stream
/// header's properties) wins, APEv2 contributes the ReplayGain items the head usually lacks,
/// and ID3v1's 30-byte Latin-1 truncations only fill what neither of the others had.
pub(crate) fn emit(
    tail: &[u8],
    base: u64,
    file_len: u64,
    t: Trailers,
    arena: &Arena,
    sink: &mut impl TagSink,
) {
    let rel = |abs: u64| -> Option<usize> { usize::try_from(abs.checked_sub(base)?).ok() };
    if t.ape > 0 {
        let end = file_len - t.v1;
        if let Some(block) = rel(end - t.ape).zip(rel(end)).and_then(|(s, e)| tail.get(s..e)) {
            pf_mp3::id3::parse_ape(block, sink);
        }
    }
    if t.v1 > 0 {
        if let Some(block) = rel(file_len - V1_LEN).and_then(|s| tail.get(s..)) {
            pf_mp3::id3::parse_v1(block, arena, sink);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::disallowed_methods)] // fixture building in tests — one-time (spec: allocation discipline)

    use super::*;
    use profluens_core::event::TagListSink;
    use profluens_core::memory::Pool;

    use crate::fixture::{ape_tag, id3v1, mp3_frames};

    #[test]
    fn finds_ape_in_front_of_id3v1() {
        let ape = ape_tag(&[("REPLAYGAIN_TRACK_GAIN", "-3.21 dB")]);
        let f = [mp3_frames(44_100, 128, 4), ape.clone(), id3v1("T", "A", "R", "2020", 7)].concat();
        let len = f.len() as u64;
        let t = measure(&f, 0, len);
        assert_eq!(t, Trailers { v1: V1_LEN, ape: ape.len() as u64 });
        assert_eq!(t.total(), V1_LEN + ape.len() as u64);

        // Same file measured through a short tail window that still ends at EOF.
        let base = len - 512;
        assert_eq!(measure(&f[(base as usize)..], base, len), t);
        // A window that does not end at EOF measures nothing.
        assert_eq!(measure(&f[..64], 0, len), Trailers::default());
    }

    #[test]
    fn a_bare_stream_has_no_trailers() {
        let f = mp3_frames(48_000, 128, 8);
        assert_eq!(measure(&f, 0, f.len() as u64), Trailers::default());
    }

    #[test]
    fn ape_alone_and_v1_alone() {
        let ape = ape_tag(&[("TITLE", "Ape Only")]);
        let f = [mp3_frames(44_100, 128, 2), ape.clone()].concat();
        let len = f.len() as u64;
        assert_eq!(measure(&f, 0, len), Trailers { v1: 0, ape: ape.len() as u64 });

        let g = [mp3_frames(44_100, 128, 2), id3v1("V1", "A", "R", "1999", 2)].concat();
        assert_eq!(measure(&g, 0, g.len() as u64), Trailers { v1: V1_LEN, ape: 0 });
    }

    #[test]
    fn emits_ape_then_v1() {
        let f = [
            mp3_frames(44_100, 128, 2),
            ape_tag(&[("TITLE", "Ape Title"), ("ALBUM", "Ape Album")]),
            id3v1("V1 Title", "V1 Artist", "V1 Album", "1997", 5),
        ]
        .concat();
        let len = f.len() as u64;
        let pool = Pool::bounded(4096, 8);
        let arena = Arena::default();
        let mut sink = TagListSink::new(&pool);
        let t = measure(&f, 0, len);
        emit(&f, 0, len, t, &arena, &mut sink);
        let tags = sink.finish();
        // APEv2 wins where both carry a key; ID3v1-only fields still land.
        assert_eq!(tags.get("TITLE"), Some("Ape Title"));
        assert_eq!(tags.get("ALBUM"), Some("Ape Album"));
        assert_eq!(tags.get("ARTIST"), Some("V1 Artist"));
        assert_eq!(tags.get("DATE"), Some("1997"));
    }

    #[test]
    fn truncation_never_panics() {
        let f = [
            mp3_frames(44_100, 128, 3),
            ape_tag(&[("TITLE", "T"), ("REPLAYGAIN_TRACK_GAIN", "0.00 dB")]),
            id3v1("t", "a", "r", "2001", 1),
        ]
        .concat();
        let pool = Pool::bounded(4096, 8);
        for n in 0..f.len() {
            let arena = Arena::default();
            let mut sink = TagListSink::new(&pool);
            let t = measure(&f[..n], 0, n as u64);
            emit(&f[..n], 0, n as u64, t, &arena, &mut sink);
        }
    }
}
