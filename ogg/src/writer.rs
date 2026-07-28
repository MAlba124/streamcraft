//! `OggWriter` — the muxer: packets for a serial number → well-formed pages (spec:
//! RFC 3533 §5 "The encapsulation process", §6 the page format).
//!
//! ## Segmentation (§5)
//! "Ogg divides each packet into 255 byte long chunks plus a final shorter chunk"
//! whose sizes are the lacing values. A lacing value of 255 "implies that a second
//! lacing value follows in the packet", and "a value of less than 255 marks the end of
//! the packet". Therefore a packet of length `L` becomes `L / 255` lacing values of
//! 255 followed by one final value of `L % 255` — and when `L` is an exact multiple of
//! 255 (§5: "A packet of 255 bytes (or a multiple of 255 bytes) is terminated by a
//! lacing value of 0") that final value is 0. A zero-length ("nil") packet is a single
//! lacing value of 0.
//!
//! ## Pages (§6)
//! A page holds at most 255 lacing values. When a packet's lacing values do not fit in
//! the current page, the page is flushed and the remainder continues on the next page
//! with the header_type CONTINUED bit set (§6, flag 0x01). A packet larger than one
//! page's worth of segments therefore spans multiple pages automatically.
//!
//! ## Granule position (§6, field 4)
//! Each page carries the granule position "after including all frames finished on this
//! page". The caller supplies a granule per *packet* ([`write_packet`]); a page's
//! granule is that of the last packet to **finish** on it. A page on which no packet
//! finishes (a packet still spilling into the next page) gets the −1 sentinel
//! [`GRANULE_NONE`](crate::page::GRANULE_NONE). Zero-copy of the codec bytes: payloads
//! are appended directly to the output.

use crate::page::{self, flags, GRANULE_NONE, MAX_SEGMENTS};

/// Muxes packets of a single logical bitstream (one serial number) into pages. For
/// several multiplexed streams, use one `OggWriter` per serial and interleave whole
/// pages in the output (§4 grouping): emit each writer's pages with [`flush`] at the
/// interleave points, or concatenate two independently-muxed streams for chaining.
pub struct OggWriter {
    serial: u32,
    /// Next page_sequence_number to emit (§6, field 6): increments per page, per
    /// logical bitstream, starting at 0.
    next_sequence: u32,
    /// Lacing values accumulated for the page currently being built (≤ 255).
    seg_table: Vec<u8>,
    /// Payload bytes accumulated for the page currently being built, parallel to
    /// `seg_table`.
    seg_payload: Vec<u8>,
    /// Granule of the most recent packet to *finish* in the pending page, or
    /// `GRANULE_NONE` if none has finished yet (a continued packet only).
    pending_granule: u64,
    /// Whether the pending page's first segment continues a packet from the previous
    /// page (sets header_type 0x01).
    pending_continued: bool,
    /// Whether the next page to be emitted is the first of the stream (sets bos).
    at_bos: bool,
    /// Set once [`finish`] has emitted the eos page; further writes are rejected.
    finished: bool,
}

/// Errors from the writer. The only failure modes are misuse (writing after `finish`);
/// segmentation itself cannot fail.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WriteError {
    /// A packet was submitted after [`OggWriter::finish`] closed the stream.
    AfterEos,
}

impl OggWriter {
    /// Start a logical bitstream with the given serial number (§6, field 5). The first
    /// page emitted will carry the bos flag.
    // COLD: one-time constructor; `seg_table`/`seg_payload` are reused across pages, cleared
    // (not reallocated) on each `flush_page`.
    #[allow(clippy::disallowed_methods)]
    pub fn new(serial: u32) -> Self {
        Self {
            serial,
            next_sequence: 0,
            seg_table: Vec::new(),
            seg_payload: Vec::new(),
            pending_granule: GRANULE_NONE,
            pending_continued: false,
            at_bos: true,
            finished: false,
        }
    }

    pub fn serial(&self) -> u32 {
        self.serial
    }

    /// Append one packet with the given granule position, flushing full pages into
    /// `out` (§5 segmentation, §6 paging). The packet's bytes are copied into page
    /// payloads; nothing is emitted until a page fills or [`flush`](Self::flush) /
    /// [`finish`](Self::finish) is called, so a partial page can still accumulate more
    /// packets (matching how real muxers pack many small packets per page).
    ///
    /// `granule` is this packet's granule position; the page on which the packet's last
    /// segment lands will report it (§6, field 4). Pass [`GRANULE_NONE`] for "unknown".
    pub fn write_packet(&mut self, out: &mut Vec<u8>, packet: &[u8], granule: u64) -> Result<(), WriteError> {
        if self.finished {
            return Err(WriteError::AfterEos);
        }

        // Split into lacing values (§5). `n_full` chunks of 255, then one final chunk
        // of `remainder` — which is 0 when the length is a multiple of 255, giving the
        // required terminating zero lacing value.
        let n_full = packet.len() / 255;
        let remainder = (packet.len() % 255) as u8;
        let mut off = 0usize;

        // Emit the 255-lacing segments.
        for _ in 0..n_full {
            self.push_segment(out, 255, &packet[off..off + 255]);
            off += 255;
        }
        // Emit the terminating (< 255, possibly 0) segment: the one that finishes the
        // packet. It lands in the currently-pending page, so stamp that page's granule
        // now — even if the 255-segments above flushed one or more full pages first
        // (those correctly carried GRANULE_NONE, no packet finishing on them).
        self.push_segment(out, remainder, &packet[off..off + remainder as usize]);
        self.pending_granule = granule;

        Ok(())
    }

    /// Push one lacing value + its bytes into the pending page, flushing the page first
    /// if the segment table is already full (255 segments, §6). A flush here happens
    /// only mid-packet (a packet longer than one page's segment table), so the flushed
    /// page finishes no packet and keeps `pending_granule` == `GRANULE_NONE`.
    fn push_segment(&mut self, out: &mut Vec<u8>, lacing: u8, bytes: &[u8]) {
        if self.seg_table.len() == MAX_SEGMENTS {
            self.flush_page(out, false);
        }
        self.seg_table.push(lacing);
        self.seg_payload.extend_from_slice(bytes);
    }

    /// Flush the pending page to `out` if it holds any segments (§6). `is_eos` sets the
    /// end-of-stream flag on this page. A no-op if there is nothing pending, *unless*
    /// `is_eos` demands an explicit (possibly empty) eos page — see [`finish`].
    pub fn flush(&mut self, out: &mut Vec<u8>) {
        if !self.seg_table.is_empty() {
            self.flush_page(out, false);
        }
    }

    /// Emit the pending page. Chooses header_type flags, granule and sequence number,
    /// then resets the pending buffers. `is_eos` sets the eos bit.
    fn flush_page(&mut self, out: &mut Vec<u8>, is_eos: bool) {
        let mut header_type = 0u8;
        if self.pending_continued {
            header_type |= flags::CONTINUED;
        }
        if self.at_bos {
            header_type |= flags::BOS;
        }
        if is_eos {
            header_type |= flags::EOS;
        }

        page::write_page(
            out,
            header_type,
            self.pending_granule,
            self.serial,
            self.next_sequence,
            &self.seg_table,
            &self.seg_payload,
        );

        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.at_bos = false;
        // If the last lacing value in this page was 255, the packet is *not* finished
        // and the next page continues it (§6, flag 0x01). Otherwise the next page
        // starts fresh.
        self.pending_continued = self.seg_table.last() == Some(&255);
        self.seg_table.clear();
        self.seg_payload.clear();
        // The next page has finished no packet yet.
        self.pending_granule = GRANULE_NONE;
    }

    /// Close the logical bitstream: flush any pending segments and guarantee a page
    /// with the eos flag set (§6, flag 0x04). If a page is pending it becomes the eos
    /// page; otherwise an empty eos page (a single nil segment) is emitted so the
    /// stream is well-terminated even when the last packet exactly filled a page.
    ///
    /// After `finish`, further [`write_packet`] calls fail with [`WriteError::AfterEos`].
    pub fn finish(&mut self, out: &mut Vec<u8>) {
        if self.finished {
            return;
        }
        if self.seg_table.is_empty() {
            // Nothing pending. Emit a nil eos page: one zero-length segment. (§6 allows
            // eos pages that are "'nil' pages … containing no content but simply a page
            // header with position information and the eos flag set".) We use a single
            // 0 lacing value so the page has a valid, empty segment table entry.
            self.seg_table.push(0);
        }
        self.flush_page(out, true);
        self.finished = true;
    }

    /// True once [`finish`] has emitted the eos page.
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

/// Convenience: mux a whole slice of `(packet, granule)` for one serial into a fresh
/// `Vec`, terminated with an eos page. Small packets are packed together into pages up
/// to the 255-segment limit, exactly as a streaming caller would get by flushing
/// rarely.
// COLD: one-shot whole-stream convenience API (not the element's per-packet path); the
// output vector is the caller's result, allocated once.
#[allow(clippy::disallowed_methods)]
pub fn mux_packets(serial: u32, packets: &[(&[u8], u64)]) -> Vec<u8> {
    let mut w = OggWriter::new(serial);
    let mut out = Vec::new();
    for &(pkt, granule) in packets {
        // Unwrap is safe: we never write after finish here.
        w.write_packet(&mut out, pkt, granule).expect("mux");
    }
    w.finish(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::PageHeader;

    /// One parsed page's fields the tests care about:
    /// `(header_type, granule, sequence, segment_table, payload)`.
    type ParsedPage = (u8, u64, u32, Vec<u8>, Vec<u8>);

    /// Parse every page in `stream`, returning them, and assert the whole stream is
    /// consumed (no trailing garbage).
    fn parse_all(stream: &[u8]) -> Vec<ParsedPage> {
        let mut pages = Vec::new();
        let mut off = 0;
        while off < stream.len() {
            let page = PageHeader::parse(&stream[off..]).expect("valid page");
            pages.push((
                page.header_type(),
                page.granule_position(),
                page.sequence(),
                page.segment_table().to_vec(),
                page.payload().to_vec(),
            ));
            off += page.len();
        }
        assert_eq!(off, stream.len(), "stream fully consumed");
        pages
    }

    #[test]
    fn single_small_packet_one_page() {
        let stream = mux_packets(7, &[(&[1, 2, 3, 4], 100)]);
        let pages = parse_all(&stream);
        // One data page (bos+eos on the same page) — first and last page of the stream.
        assert_eq!(pages.len(), 1);
        let (ht, gran, seq, table, payload) = &pages[0];
        assert_eq!(*ht, flags::BOS | flags::EOS);
        assert_eq!(*gran, 100);
        assert_eq!(*seq, 0);
        assert_eq!(table, &vec![4]);
        assert_eq!(payload, &vec![1, 2, 3, 4]);
    }

    #[test]
    fn packet_length_multiple_of_255_gets_terminating_zero() {
        // 510 bytes = 2 × 255 → lacing [255, 255, 0]; the trailing 0 terminates it.
        let pkt: Vec<u8> = (0..510u32).map(|i| i as u8).collect();
        let stream = mux_packets(1, &[(&pkt, 5)]);
        let pages = parse_all(&stream);
        assert_eq!(pages.len(), 1);
        let (_ht, _gran, _seq, table, payload) = &pages[0];
        assert_eq!(table, &vec![255, 255, 0]);
        assert_eq!(payload.len(), 510);
        assert_eq!(payload, &pkt);
    }

    #[test]
    fn zero_length_packet_is_single_zero_lacing() {
        let stream = mux_packets(1, &[(&[], 0)]);
        let pages = parse_all(&stream);
        assert_eq!(pages.len(), 1);
        let (_ht, _gran, _seq, table, payload) = &pages[0];
        assert_eq!(table, &vec![0]);
        assert!(payload.is_empty());
    }

    #[test]
    fn packet_spanning_multiple_pages_sets_continued() {
        // A packet needing more than 255 segments forces a second page with the
        // CONTINUED flag. 255 * 255 = 65025 bytes fills one page's segment table
        // exactly with 255-lacings and no terminator, so the packet continues.
        let pkt: Vec<u8> = (0..70_000u32).map(|i| (i * 7) as u8).collect();
        let stream = mux_packets(9, &[(&pkt, 42)]);
        let pages = parse_all(&stream);
        assert!(pages.len() >= 2, "packet must span >1 page, got {}", pages.len());

        // First page: bos, full 255-segment table all 255s, no packet finishes → NONE.
        let (ht0, gran0, seq0, table0, _) = &pages[0];
        assert_eq!(*ht0 & flags::BOS, flags::BOS);
        assert_eq!(*ht0 & flags::CONTINUED, 0, "bos page is not continued");
        assert_eq!(table0.len(), MAX_SEGMENTS);
        assert!(table0.iter().all(|&l| l == 255));
        assert_eq!(*gran0, GRANULE_NONE, "no packet finishes on page 0");
        assert_eq!(*seq0, 0);

        // Middle/last continuation pages carry the CONTINUED flag.
        let (ht1, _gran1, seq1, _t1, _p1) = &pages[1];
        assert_eq!(*ht1 & flags::CONTINUED, flags::CONTINUED, "page 1 continues packet");
        assert_eq!(*seq1, 1);

        // Reassemble the payload across all pages and compare to the original packet.
        let mut reassembled = Vec::new();
        for (_ht, _g, _s, _t, payload) in &pages {
            reassembled.extend_from_slice(payload);
        }
        assert_eq!(reassembled, pkt);

        // The granule ends up on the final page (where the packet finishes).
        let last = pages.last().unwrap();
        assert_eq!(last.1, 42, "granule on the finishing page");
        assert_eq!(last.0 & flags::EOS, flags::EOS, "last page is eos");
    }

    #[test]
    fn many_small_packets_pack_into_one_segment_table() {
        // 200 one-byte packets → 200 lacing values of 1, all in a single page (≤ 255).
        let pkts: Vec<Vec<u8>> = (0..200u32).map(|i| vec![i as u8]).collect();
        let refs: Vec<(&[u8], u64)> = pkts.iter().enumerate().map(|(i, p)| (p.as_slice(), i as u64)).collect();
        let stream = mux_packets(3, &refs);
        let pages = parse_all(&stream);
        // 200 segments fit in one page (max 255); finish adds nothing new because the
        // page is pending → it just gets the eos flag. So exactly one page.
        assert_eq!(pages.len(), 1);
        let (ht, gran, _seq, table, _payload) = &pages[0];
        assert_eq!(table.len(), 200);
        assert!(table.iter().all(|&l| l == 1));
        assert_eq!(*ht, flags::BOS | flags::EOS);
        // Granule of the last packet to finish (packet 199).
        assert_eq!(*gran, 199);
    }

    #[test]
    fn writing_after_finish_is_rejected() {
        let mut w = OggWriter::new(1);
        let mut out = Vec::new();
        w.write_packet(&mut out, &[1], 0).unwrap();
        w.finish(&mut out);
        assert_eq!(w.write_packet(&mut out, &[2], 1), Err(WriteError::AfterEos));
    }

    #[test]
    fn exactly_255_segments_then_more_packets_spills_pages() {
        // 255 one-byte packets fill the segment table exactly; a 256th packet must go
        // on a fresh page (not continued — it is a new packet).
        let pkts: Vec<Vec<u8>> = (0..256u32).map(|i| vec![i as u8]).collect();
        let refs: Vec<(&[u8], u64)> = pkts.iter().enumerate().map(|(i, p)| (p.as_slice(), i as u64)).collect();
        let stream = mux_packets(4, &refs);
        let pages = parse_all(&stream);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].3.len(), 255); // first page: 255 lacing values
        // Page 0 finishes packet 254 (its last full segment) → granule 254.
        assert_eq!(pages[0].1, 254);
        assert_eq!(pages[0].0 & flags::CONTINUED, 0);
        // Page 1 is NOT continued: packet 254 ended with a <255 lacing on page 0.
        assert_eq!(pages[1].0 & flags::CONTINUED, 0);
        assert_eq!(pages[1].3, vec![1]); // one segment: packet 255
        assert_eq!(pages[1].1, 255);
    }
}
