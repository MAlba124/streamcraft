//! `OggReader` — the demuxer: a byte stream → verified pages → reassembled packets
//! (spec: RFC 3533 §5 encapsulation, §6 page format).
//!
//! ## What it does
//! - Scans the input for the `OggS` capture pattern, parses each page, and **verifies
//!   its CRC** ([`crate::page::PageHeader::parse`]).
//! - Reassembles packets from lacing values, across page boundaries: a run of 255
//!   lacing values continues a packet, a value < 255 ends it (§5). A page flagged
//!   CONTINUED (§6, flag 0x01) appends its first segment run to the partial packet held
//!   for that serial.
//! - Tracks one reassembly state **per bitstream serial number**, so concurrently
//!   multiplexed ("grouped", §4) streams are demuxed correctly; each emitted packet
//!   carries its serial, granule position, and bos/eos.
//!
//! ## Robustness (§6)
//! The capture pattern exists so a decoder can "regain synchronisation after parsing a
//! corrupted stream". On any bad byte — wrong pattern, bad version, or a CRC mismatch —
//! the reader advances one byte and rescans for the next `OggS`. It never panics on
//! malformed input (the decoder-safety P0). Truncated tails are simply held until more
//! bytes arrive (the whole-buffer [`OggReader::read`] path) or reported as leftover.
//!
//! ## API shapes
//! - [`OggReader`] is a **push/pull streaming** demuxer: feed bytes with
//!   [`push`](OggReader::push), drain ready packets with [`next_packet`](OggReader::next_packet).
//!   Suited to network/pipe input arriving in arbitrary chunks.
//! - [`demux_all`] is a one-shot over a complete in-memory stream, returning every
//!   packet. Both share the same page/packet engine.

use std::collections::HashMap;
use std::collections::VecDeque;

use crate::page::{PageError, PageHeader, CAPTURE_PATTERN, HEADER_FIXED_LEN};

/// A reassembled packet handed out by the demuxer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Packet {
    /// Which logical bitstream this packet belongs to (§6, field 5).
    pub serial: u32,
    /// The page granule position of the page on which this packet **finished** (§6,
    /// field 4). `u64::MAX` ([`GRANULE_NONE`](crate::page::GRANULE_NONE)) if that page
    /// declared no finishing packet — unusual for a finishing packet, but passed
    /// through verbatim rather than invented.
    pub granule: u64,
    /// True if this packet is the first packet of its logical bitstream — it began on a
    /// bos page (§6, flag 0x02). By convention the bos page holds exactly the codec's
    /// identification packet.
    pub bos: bool,
    /// True if this packet finished on the eos page of its logical bitstream (§6, flag
    /// 0x04) — i.e. the last packet of the stream.
    pub eos: bool,
    /// The packet payload (concatenated segment bytes).
    pub data: Vec<u8>,
}

/// Per-serial packet reassembly state.
struct StreamState {
    /// Bytes of a packet still being assembled (segments seen so far whose run has not
    /// yet been terminated by a < 255 lacing value).
    partial: Vec<u8>,
    /// True while `partial` holds an unfinished packet spilling into the next page.
    in_progress: bool,
    /// True until the first packet of this stream has been emitted (so we can tag it
    /// `bos`). Set when the bos page is seen.
    pending_bos: bool,
    /// Set once the eos page has been processed for this serial.
    seen_eos: bool,
}

impl StreamState {
    fn new() -> Self {
        Self {
            partial: Vec::new(),
            in_progress: false,
            pending_bos: false,
            seen_eos: false,
        }
    }
}

/// A streaming Ogg demuxer. Feed bytes with [`push`](Self::push); pull decoded packets
/// with [`next_packet`](Self::next_packet). Holds one [`StreamState`] per serial so
/// multiplexed streams reassemble independently.
pub struct OggReader {
    /// Unconsumed input: bytes pushed but not yet forming a complete page.
    buf: Vec<u8>,
    /// Read cursor into `buf` (bytes before it are consumed; compacted lazily).
    pos: usize,
    /// Reassembly state keyed by bitstream serial number.
    streams: HashMap<u32, StreamState>,
    /// Ready packets awaiting the caller.
    ready: VecDeque<Packet>,
    /// Diagnostics: count of resync events (pattern/version/CRC failures skipped).
    resyncs: u64,
    /// Diagnostics: total pages successfully parsed.
    pages_parsed: u64,
}

impl OggReader {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
            streams: HashMap::new(),
            ready: VecDeque::new(),
            resyncs: 0,
            pages_parsed: 0,
        }
    }

    /// Feed more input. Parses as many whole pages as are now available, queueing any
    /// completed packets for [`next_packet`](Self::next_packet). Partial trailing bytes
    /// are retained for the next `push`.
    pub fn push(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
        self.drive();
    }

    /// Signal end of input: no more bytes will be pushed. Any bytes left in the buffer
    /// that never formed a valid page are dropped (they are trailing garbage or a
    /// truncated final page). Already-reassembled packets remain available.
    ///
    /// Returns the number of unparsed leftover bytes discarded (0 on a clean stream end
    /// at a page boundary).
    pub fn finish(&mut self) -> usize {
        self.drive();
        let leftover = self.buf.len() - self.pos;
        self.buf.clear();
        self.pos = 0;
        leftover
    }

    /// Pull the next ready packet, or `None` if none is currently assembled.
    pub fn next_packet(&mut self) -> Option<Packet> {
        self.ready.pop_front()
    }

    /// Number of resynchronisation events so far (bad capture pattern, version, or CRC).
    pub fn resync_count(&self) -> u64 {
        self.resyncs
    }

    /// Number of pages successfully parsed so far.
    pub fn pages_parsed(&self) -> u64 {
        self.pages_parsed
    }

    /// Parse all currently-available pages from `buf[pos..]`.
    fn drive(&mut self) {
        loop {
            // Nothing more can be a page if fewer than the fixed header remain, unless
            // there is enough to at least try (parse reports NeedMore). Try to parse at
            // the cursor.
            let remaining = &self.buf[self.pos..];
            if remaining.len() < HEADER_FIXED_LEN {
                break; // wait for more bytes
            }
            match PageHeader::parse(remaining) {
                Ok(page) => {
                    let len = page.len();
                    // `page` borrows `self.buf`, so drive the reassembly through a free
                    // function over the disjoint `streams`/`ready` fields rather than a
                    // `&mut self` method (which the borrow checker would reject).
                    handle_page(&mut self.streams, &mut self.ready, &page);
                    self.pos += len;
                    self.pages_parsed += 1;
                }
                Err(PageError::NeedMoreHeader) | Err(PageError::NeedMoreBody) => {
                    // A valid-looking page start but not all bytes are here yet. Two
                    // sub-cases: if the capture pattern is actually correct we wait; if
                    // NeedMore came from a mis-synced position it will keep failing, so
                    // only wait when byte 0 really is a capture pattern. Otherwise
                    // resync.
                    if remaining.len() >= 4 && remaining[0..4] == CAPTURE_PATTERN {
                        break; // genuine partial page: wait for more data
                    }
                    self.resync_one();
                }
                Err(PageError::BadCapturePattern)
                | Err(PageError::BadVersion(_))
                | Err(PageError::BadCrc { .. }) => {
                    self.resync_one();
                }
            }
            // Keep the buffer from growing unbounded once a good chunk is consumed.
            self.maybe_compact();
        }
        self.maybe_compact();
    }

    /// Advance past one byte and scan forward to the next capture pattern, so the next
    /// `parse` attempt starts at a plausible page boundary (§6 resync). Counts one
    /// resync event. On a partial buffer with no further `OggS`, leaves the cursor at
    /// the last position that could still begin a pattern.
    fn resync_one(&mut self) {
        self.resyncs += 1;
        // Move at least one byte forward, then find the next 'O' of "OggS".
        let start = self.pos + 1;
        let hay = &self.buf[start..];
        match find_capture(hay) {
            Some(rel) => self.pos = start + rel,
            None => {
                // No full capture pattern remains. Keep up to the last 3 bytes (a
                // pattern could be split across the next push); drop the rest.
                let keep_from = self.buf.len().saturating_sub(CAPTURE_PATTERN.len() - 1);
                self.pos = keep_from.max(start);
            }
        }
    }

    /// Reclaim consumed bytes from the front of `buf`, bounding memory on long streams
    /// without turning a single large `push` into O(n²) work.
    ///
    /// A `drain(..pos)` memmoves the *unconsumed tail* down by `pos`. If we did that
    /// every 64 KiB of progress through a fully-buffered 256 MiB input, we would memmove
    /// ~256 MiB thousands of times — the whole demux would be O(n²) (this is exactly why
    /// a naive threshold is a trap). So we only compact when it is cheap or necessary:
    ///
    /// - everything so far is consumed (`pos == len`): just clear — O(1), the common
    ///   end-of-`drive` case for a big push;
    /// - otherwise compact only when the unconsumed tail we would move is *small*
    ///   (≤ `MAX_MOVE`) yet the consumed prefix is large — the streaming case, where the
    ///   tail is a partial page. A huge already-buffered tail is left in place (its bytes
    ///   are walked, not copied), so total work stays O(n).
    fn maybe_compact(&mut self) {
        const CONSUMED_TRIGGER: usize = 64 * 1024;
        const MAX_MOVE: usize = 64 * 1024;
        if self.pos == 0 {
            return;
        }
        let tail = self.buf.len() - self.pos;
        if tail == 0 {
            self.buf.clear();
            self.pos = 0;
        } else if self.pos >= CONSUMED_TRIGGER && tail <= MAX_MOVE {
            self.buf.drain(..self.pos);
            self.pos = 0;
        }
    }
}

impl Default for OggReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Feed one verified page's segments into the per-serial reassembly, queueing any
/// completed packets onto `ready`. A free function (not a `&mut self` method) because
/// `page` borrows the reader's input buffer while `streams`/`ready` are disjoint —
/// passing them explicitly is what lets the borrow checker see the disjointness.
fn handle_page(
    streams: &mut HashMap<u32, StreamState>,
    ready: &mut VecDeque<Packet>,
    page: &PageHeader<'_>,
) {
    let serial = page.serial();
    let granule = page.granule_position();
    let continued = page.is_continued();
    let bos = page.is_bos();
    let eos = page.is_eos();

    let state = streams.entry(serial).or_insert_with(StreamState::new);
    if bos {
        state.pending_bos = true;
    }

    // If this page does *not* continue a packet but we were mid-packet, the previous
    // partial is orphaned (a page was lost — §6 sequence numbers exist to notice this):
    // drop the fragment and start fresh, rather than splicing across a gap and
    // corrupting the packet.
    if !continued && state.in_progress {
        state.partial.clear();
        state.in_progress = false;
    }

    // Walk the segment table, accumulating packet bytes. A lacing value of 255 continues
    // the current packet; < 255 ends it after that many bytes (§5). The page payload is
    // the concatenation of all segment bytes in order.
    let table = page.segment_table();
    let payload = page.payload();
    let mut off = 0usize;
    // Index of the last lacing value, to know if the final packet on the page is
    // complete (last lacing < 255) or spills to the next page (last lacing == 255).
    let last_idx = table.len().wrapping_sub(1);

    for (i, &lacing) in table.iter().enumerate() {
        let seg = &payload[off..off + lacing as usize];
        off += lacing as usize;
        state.partial.extend_from_slice(seg);
        state.in_progress = true;

        if lacing < 255 {
            // Packet complete. A completed packet — whether or not it is the last on the
            // page — belongs to this page, so it takes this page's granule (§6: granule
            // covers "all frames finished on this page").
            let data = std::mem::take(&mut state.partial);
            state.in_progress = false;
            let is_last_on_page = i == last_idx;
            let packet = Packet {
                serial,
                granule,
                bos: state.pending_bos,
                // Tag eos only for the packet completing on the eos page's *last*
                // segment — the stream's final packet. A mid-page packet on an eos page
                // is not "the" eos packet. For the common one-packet-per-eos-page case
                // this is exactly right.
                eos: eos && is_last_on_page,
                data,
            };
            // Once the first packet is emitted, later packets are not bos.
            state.pending_bos = false;
            ready.push_back(packet);
        }
        // lacing == 255: packet continues into the next segment / page; keep
        // accumulating in `partial` with in_progress = true.
    }

    if eos {
        state.seen_eos = true;
        // A trailing in-progress packet on an eos page can never be completed (nothing
        // continues it), so drop the fragment to avoid leaking it.
        if state.in_progress {
            state.partial.clear();
            state.in_progress = false;
        }
    }
}

/// Find the byte offset of the next `OggS` capture pattern in `hay`, or `None`. A plain
/// forward scan; the capture pattern is short and rare enough that this is not a
/// bottleneck (and the whole point is robustness, not speed, on the error path).
fn find_capture(hay: &[u8]) -> Option<usize> {
    if hay.len() < CAPTURE_PATTERN.len() {
        return None;
    }
    hay.windows(CAPTURE_PATTERN.len())
        .position(|w| w == CAPTURE_PATTERN)
}

/// One-shot demux of a complete in-memory Ogg stream into all its packets, in stream
/// order (spec §5). Convenience over the streaming [`OggReader`]; resyncs past garbage
/// exactly the same way and never panics.
pub fn demux_all(stream: &[u8]) -> Vec<Packet> {
    let mut reader = OggReader::new();
    reader.push(stream);
    reader.finish();
    let mut out = Vec::new();
    while let Some(p) = reader.next_packet() {
        out.push(p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::mux_packets;

    #[test]
    fn find_capture_locates_pattern() {
        assert_eq!(find_capture(b"OggS"), Some(0));
        assert_eq!(find_capture(b"xxOggS"), Some(2));
        assert_eq!(find_capture(b"no pattern here"), None);
        assert_eq!(find_capture(b"Ogg"), None); // too short
        assert_eq!(find_capture(b"OggOggS"), Some(3)); // false start then real one
    }

    #[test]
    fn roundtrip_single_packet() {
        let stream = mux_packets(7, &[(&[1, 2, 3, 4], 100)]);
        let packets = demux_all(&stream);
        assert_eq!(packets.len(), 1);
        assert_eq!(packets[0].serial, 7);
        assert_eq!(packets[0].granule, 100);
        assert!(packets[0].bos);
        assert!(packets[0].eos);
        assert_eq!(packets[0].data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn streaming_push_in_tiny_chunks() {
        // Feeding the stream one byte at a time must produce the same packets: the
        // reader holds partial pages until complete.
        let src: Vec<Vec<u8>> = vec![vec![9; 300], vec![], vec![1, 2], vec![7; 600]];
        let refs: Vec<(&[u8], u64)> = src.iter().enumerate().map(|(i, p)| (p.as_slice(), i as u64)).collect();
        let stream = mux_packets(1, &refs);

        let mut reader = OggReader::new();
        let mut got = Vec::new();
        for &b in &stream {
            reader.push(&[b]);
            while let Some(p) = reader.next_packet() {
                got.push(p);
            }
        }
        reader.finish();
        while let Some(p) = reader.next_packet() {
            got.push(p);
        }

        assert_eq!(got.len(), src.len());
        for (i, p) in got.iter().enumerate() {
            assert_eq!(p.data, src[i], "packet {i}");
        }
    }

    #[test]
    fn bos_eos_flags_on_first_and_last_packet() {
        let src: Vec<Vec<u8>> = (0..5u32).map(|i| vec![i as u8; 10]).collect();
        let refs: Vec<(&[u8], u64)> = src.iter().enumerate().map(|(i, p)| (p.as_slice(), i as u64 * 10)).collect();
        let stream = mux_packets(2, &refs);
        let packets = demux_all(&stream);
        assert_eq!(packets.len(), 5);
        assert!(packets.first().unwrap().bos);
        assert!(!packets.first().unwrap().eos);
        assert!(packets.last().unwrap().eos);
        assert!(!packets.last().unwrap().bos);
        for p in &packets[1..4] {
            assert!(!p.bos);
            assert!(!p.eos);
        }
    }
}
