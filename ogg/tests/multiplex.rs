//! Concurrent multiplexing ("grouping", §4): two logical bitstreams interleaved at the
//! page level demux into two correct, independent packet streams (spec: RFC 3533 §4).
//!
//! We build each stream's pages with a per-serial [`OggWriter`], flushing per packet so
//! the pages are whole and can be interleaved, then splice the pages together in a
//! round-robin order — exactly how a real muxer lays out grouped streams. The demuxer,
//! which keys reassembly by serial number, must separate them again.

use sc_ogg::{demux_all, OggWriter, Packet};

/// SplitMix64 for reproducible payloads.
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

/// A logical stream's packets as `(bytes, granule)` pairs.
type PacketSet = Vec<(Vec<u8>, u64)>;

/// Split a muxed single-serial stream back into its individual page byte-ranges, so we
/// can re-interleave whole pages from two streams. Uses the public parser via
/// `demux`-independent page walking through the crate's `PageHeader`.
fn split_pages(stream: &[u8]) -> Vec<Vec<u8>> {
    // Re-parse to page boundaries using the reader's own page view would be ideal, but
    // the public API exposes `PageHeader`. Walk it.
    use sc_ogg::PageHeader;
    let mut pages = Vec::new();
    let mut off = 0;
    while off < stream.len() {
        let page = PageHeader::parse(&stream[off..]).expect("valid page in own output");
        let len = page.len();
        pages.push(stream[off..off + len].to_vec());
        off += len;
    }
    pages
}

/// Produce the pages of one logical bitstream, one page per packet, so they can be
/// freely interleaved with another stream's pages. We flush *between* packets but not
/// after the last one, so `finish()` puts the eos flag on the final data page rather
/// than emitting a separate nil eos page — which keeps the expected model (eos on the
/// last data packet) exact. A multi-page packet still spans several of these pages.
fn stream_pages(serial: u32, packets: &[(Vec<u8>, u64)]) -> Vec<Vec<u8>> {
    let mut w = OggWriter::new(serial);
    let mut out = Vec::new();
    for (i, (pkt, gran)) in packets.iter().enumerate() {
        if i > 0 {
            w.flush(&mut out); // close the previous packet's page before starting a new one
        }
        w.write_packet(&mut out, pkt, *gran).unwrap();
    }
    w.finish(&mut out); // flushes the last packet's page with eos set
    split_pages(&out)
}

/// The packets we expect back for a serial, tagged with bos on the first and eos on the
/// last (the writer flags them; the demuxer surfaces them).
fn expected(serial: u32, packets: &[(Vec<u8>, u64)]) -> Vec<Packet> {
    packets
        .iter()
        .enumerate()
        .map(|(i, (data, gran))| Packet {
            serial,
            granule: *gran,
            bos: i == 0,
            eos: i == packets.len() - 1,
            data: data.clone(),
        })
        .collect()
}

#[test]
fn two_streams_interleaved_demux_independently() {
    // Stream A: a handful of medium packets. Stream B: different sizes, including a
    // multi-page packet, so the interleave has B-continuation pages between A pages.
    let a: PacketSet = vec![
        (payload(0xA1, 40), 10),
        (payload(0xA2, 255), 20),
        (payload(0xA3, 300), 30),
        (payload(0xA4, 5), 40),
    ];
    let b: PacketSet = vec![
        (payload(0xB1, 66_000), 100), // spans multiple pages
        (payload(0xB2, 10), 200),
        (payload(0xB3, 0), 300), // nil packet
    ];

    let a_serial = 0x1111_1111;
    let b_serial = 0x2222_2222;
    let a_pages = stream_pages(a_serial, &a);
    let b_pages = stream_pages(b_serial, &b);

    // Interleave: per §4, all bos pages come first, then a round-robin of the rest.
    let mut stream = Vec::new();
    // bos pages first (page 0 of each).
    stream.extend_from_slice(&a_pages[0]);
    stream.extend_from_slice(&b_pages[0]);
    // Round-robin the remaining pages.
    let mut ai = 1;
    let mut bi = 1;
    while ai < a_pages.len() || bi < b_pages.len() {
        if ai < a_pages.len() {
            stream.extend_from_slice(&a_pages[ai]);
            ai += 1;
        }
        if bi < b_pages.len() {
            stream.extend_from_slice(&b_pages[bi]);
            bi += 1;
        }
    }

    let got = demux_all(&stream);
    // Separate by serial, preserving order within each.
    let got_a: Vec<Packet> = got.iter().filter(|p| p.serial == a_serial).cloned().collect();
    let got_b: Vec<Packet> = got.iter().filter(|p| p.serial == b_serial).cloned().collect();

    assert_eq!(got_a, expected(a_serial, &a), "stream A packets");
    assert_eq!(got_b, expected(b_serial, &b), "stream B packets");
}

#[test]
fn three_streams_round_robin() {
    // Three serials, each a few packets, interleaved page-by-page. All three must come
    // back complete and correctly attributed.
    let streams: Vec<(u32, PacketSet)> = vec![
        (10, (0..6).map(|i| (payload(10_000 + i, 50 + i as usize * 7), i)).collect()),
        (20, (0..4).map(|i| (payload(20_000 + i, 255 * (i as usize + 1)), i * 2)).collect()),
        (30, (0..5).map(|i| (payload(30_000 + i, i as usize % 3), i * 3)).collect()),
    ];

    let paged: Vec<(u32, Vec<Vec<u8>>)> = streams
        .iter()
        .map(|(s, pkts)| (*s, stream_pages(*s, pkts)))
        .collect();

    // bos pages first.
    let mut stream = Vec::new();
    for (_s, pages) in &paged {
        stream.extend_from_slice(&pages[0]);
    }
    // Round-robin the rest.
    let mut idx = vec![1usize; paged.len()];
    loop {
        let mut progressed = false;
        for (i, (_s, pages)) in paged.iter().enumerate() {
            if idx[i] < pages.len() {
                stream.extend_from_slice(&pages[idx[i]]);
                idx[i] += 1;
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }

    let got = demux_all(&stream);
    for (serial, pkts) in &streams {
        let got_s: Vec<Packet> = got.iter().filter(|p| p.serial == *serial).cloned().collect();
        assert_eq!(got_s, expected(*serial, pkts), "stream {serial}");
    }
}
