//! Robustness: the demuxer recovers at the next valid page after garbage or truncation,
//! and never panics on malformed input (spec: RFC 3533 §6 — the capture pattern exists
//! so a decoder can "regain synchronisation after parsing a corrupted stream"; and the
//! decoder-safety P0: a crash on any bitstream is unacceptable).

use pf_ogg::{demux_all, mux_packets, OggReader};

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
fn payload(seed: u64, n: usize) -> Vec<u8> {
    let mut r = Rng(seed);
    (0..n).map(|_| r.next_u64() as u8).collect()
}

/// A small, valid multi-packet stream reused across the resync tests.
fn good_stream() -> (Vec<u8>, Vec<Vec<u8>>) {
    let pkts: Vec<Vec<u8>> = vec![
        payload(1, 30),
        payload(2, 255),
        payload(3, 500),
        payload(4, 7),
    ];
    let refs: Vec<(&[u8], u64)> = pkts.iter().enumerate().map(|(i, p)| (p.as_slice(), i as u64)).collect();
    (mux_packets(1, &refs), pkts)
}

#[test]
fn leading_garbage_is_skipped() {
    let (stream, pkts) = good_stream();
    let mut junk = vec![0u8; 137];
    junk.iter_mut().enumerate().for_each(|(i, b)| *b = (i as u8).wrapping_mul(53) ^ 0x5A);
    // Make sure the junk contains no accidental "OggS".
    junk.retain(|_| true);
    let mut input = junk.clone();
    input.extend_from_slice(&stream);

    let got = demux_all(&input);
    let datas: Vec<Vec<u8>> = got.into_iter().map(|p| p.data).collect();
    assert_eq!(datas, pkts, "packets recovered after leading garbage");
}

#[test]
fn garbage_containing_fake_oggs_still_recovers() {
    // Adversarial: garbage that includes the bytes "OggS" but is not a valid page. The
    // reader must reject it on version/CRC and keep scanning to the real page.
    let (stream, pkts) = good_stream();
    let mut input = Vec::new();
    input.extend_from_slice(b"....OggS not a real page, bad version and crc....");
    input.extend_from_slice(b"OggS\xFF\xFF\xFF\xFF garbage after a fake capture pattern");
    input.extend_from_slice(&stream);

    let got = demux_all(&input);
    let datas: Vec<Vec<u8>> = got.into_iter().map(|p| p.data).collect();
    assert_eq!(datas, pkts, "recovered past fake OggS markers");
}

#[test]
fn corruption_in_the_middle_drops_one_page_recovers_after() {
    // Flip bytes inside the *second* page's payload so its CRC fails. That page is
    // skipped; the reader resyncs to the third page. Packets wholly on the surviving
    // pages come through; the packet on the corrupted page is lost, not fatal.
    let pkts: Vec<Vec<u8>> = (0..6u32).map(|i| payload(100 + i as u64, 60)).collect();
    // One packet per page so a corrupted page loses exactly one packet. Flush between
    // packets (not after the last) so `finish()` puts eos on the final data page rather
    // than a separate nil eos page — keeping "N packets, one per page".
    let mut w = pf_ogg::OggWriter::new(1);
    let mut stream = Vec::new();
    for (i, p) in pkts.iter().enumerate() {
        if i > 0 {
            w.flush(&mut stream);
        }
        w.write_packet(&mut stream, p, i as u64).unwrap();
    }
    w.finish(&mut stream);

    // Find the second page and corrupt a byte in its body (not the capture pattern, so
    // the CRC check is what rejects it).
    use pf_ogg::PageHeader;
    let first = PageHeader::parse(&stream).unwrap();
    let second_off = first.len();
    let second = PageHeader::parse(&stream[second_off..]).unwrap();
    // Corrupt the last payload byte of the second page.
    let corrupt_at = second_off + second.len() - 1;
    stream[corrupt_at] ^= 0xFF;

    let mut reader = OggReader::new();
    reader.push(&stream);
    reader.finish();
    let mut got = Vec::new();
    while let Some(p) = reader.next_packet() {
        got.push(p.data);
    }

    // The packet carried on the corrupted second page is missing; all others survive
    // in order.
    assert!(reader.resync_count() >= 1, "a resync must have occurred");
    let mut expected: Vec<Vec<u8>> = pkts.clone();
    expected.remove(1); // second page carried packet index 1
    assert_eq!(got, expected, "all packets except the corrupted page's survive");
}

#[test]
fn truncated_final_page_is_dropped_not_fatal() {
    // Cut the stream in the middle of the last page. The completed earlier packets must
    // still be recovered; the partial tail is silently dropped by finish().
    let (stream, pkts) = good_stream();
    let cut = stream.len() - 3; // chop the tail
    let truncated = &stream[..cut];

    let mut reader = OggReader::new();
    reader.push(truncated);
    let leftover = reader.finish();
    let mut got = Vec::new();
    while let Some(p) = reader.next_packet() {
        got.push(p.data);
    }
    // We lose whatever was on the truncated final page but recover the rest, and
    // finish() reports the dropped byte count.
    assert!(leftover > 0, "truncated tail is reported as leftover");
    assert!(got.len() < pkts.len(), "the truncated page's packet is not emitted");
    assert_eq!(got.as_slice(), &pkts[..got.len()], "recovered packets are a correct prefix");
}

#[test]
fn interior_garbage_between_pages() {
    // Insert random bytes between two good pages; recovery continues at the next page.
    let pkts: Vec<Vec<u8>> = (0..5u32).map(|i| payload(200 + i as u64, 80)).collect();
    // Flush between packets (not after the last) → one packet per page, eos on the
    // final data page, no separate nil eos page.
    let mut w = pf_ogg::OggWriter::new(3);
    let mut good = Vec::new();
    for (i, p) in pkts.iter().enumerate() {
        if i > 0 {
            w.flush(&mut good);
        }
        w.write_packet(&mut good, p, i as u64).unwrap();
    }
    w.finish(&mut good);

    use pf_ogg::PageHeader;
    // Split into pages, then rejoin with junk between each.
    let mut pages = Vec::new();
    let mut off = 0;
    while off < good.len() {
        let page = PageHeader::parse(&good[off..]).unwrap();
        pages.push(good[off..off + page.len()].to_vec());
        off += page.len();
    }
    let mut input = Vec::new();
    for (i, page) in pages.iter().enumerate() {
        input.extend_from_slice(page);
        // Junk between pages (avoid emitting a real capture pattern by construction).
        input.extend_from_slice(&[0xDE, 0xAD, i as u8, 0x00, 0x42, 0x13]);
    }

    let got = demux_all(&input);
    let datas: Vec<Vec<u8>> = got.into_iter().map(|p| p.data).collect();
    assert_eq!(datas, pkts, "all packets recovered despite inter-page junk");
}

#[test]
fn never_panics_on_random_input() {
    // Fuzz-lite: feed a lot of random byte strings (some seeded with "OggS" fragments)
    // and assert only that we never panic and terminate. Any packets produced are fine.
    let mut r = Rng(0xF0F0_1234);
    for _ in 0..2000 {
        let len = (r.next_u64() % 512) as usize;
        let mut data: Vec<u8> = (0..len).map(|_| r.next_u64() as u8).collect();
        // Sprinkle capture patterns to drive the resync path hard.
        if len > 8 {
            let at = (r.next_u64() as usize) % (len - 4);
            data[at..at + 4].copy_from_slice(b"OggS");
        }
        // Two entry points: one-shot and byte-at-a-time streaming.
        let _ = demux_all(&data);

        let mut reader = OggReader::new();
        for &b in &data {
            reader.push(&[b]);
            while reader.next_packet().is_some() {}
        }
        reader.finish();
        while reader.next_packet().is_some() {}
    }
}
