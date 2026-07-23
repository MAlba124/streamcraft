//! The correctness gate: mux → demux recovers the exact packets, in order, with the
//! right bos/eos and granule (spec: RFC 3533 §5 encapsulation, §6 pages).
//!
//! Table- and property-driven over generated packet sets: sizes including 0, exactly
//! 255, multiples of 255 (which need a trailing zero lacing value), packets spanning
//! many pages, and many tiny packets filling a segment table. Each set is muxed with
//! [`OggWriter`] and demuxed with [`OggReader`]; every packet, its order, and its
//! bos/eos flags are compared.

use sc_ogg::{demux_all, mux_packets, OggReader, OggWriter, Packet};

/// Deterministic PRNG (SplitMix64) so generated packet sets are reproducible.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Fill `n` bytes with a serial/index-seeded pattern so different packets have
/// different, checkable contents.
fn payload(seed: u64, n: usize) -> Vec<u8> {
    let mut r = Rng(seed);
    (0..n).map(|_| r.next_u64() as u8).collect()
}

/// Mux `packets` for one serial, demux, and assert exact recovery in order with the
/// expected bos on the first and eos on the last packet.
fn check_roundtrip(serial: u32, packets: &[Vec<u8>]) {
    let refs: Vec<(&[u8], u64)> = packets
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_slice(), i as u64 + 1))
        .collect();
    let stream = mux_packets(serial, &refs);
    let got = demux_all(&stream);

    assert_eq!(got.len(), packets.len(), "packet count (serial {serial})");
    for (i, (g, want)) in got.iter().zip(packets).enumerate() {
        assert_eq!(g.serial, serial, "serial on packet {i}");
        assert_eq!(&g.data, want, "packet {i} bytes (len {})", want.len());
    }
    if !packets.is_empty() {
        assert!(got.first().unwrap().bos, "first packet is bos");
        assert!(got.last().unwrap().eos, "last packet is eos");
        // No interior packet is bos; interior packets are not eos.
        for p in &got[1..] {
            assert!(!p.bos, "only the first packet may be bos");
        }
        for p in &got[..got.len() - 1] {
            assert!(!p.eos, "only the last packet may be eos");
        }
    }
}

#[test]
fn lacing_edge_case_sizes() {
    // Each size individually as a single-packet stream (each exercises its lacing).
    let sizes = [
        0usize, 1, 2, 100, 254, 255, 256, 509, 510, 511, 764, 765, 766, 65025, 65026, 70_000,
        200_000,
    ];
    for (i, &sz) in sizes.iter().enumerate() {
        check_roundtrip(1000 + i as u32, &[payload(sz as u64, sz)]);
    }
}

#[test]
fn multiples_of_255_need_trailing_zero() {
    // 0, 255, 510, 765, … all end on a 255-lacing and require a terminating 0 lacing
    // value (§5). Recovering the exact length proves the zero terminator was written
    // and consumed, not treated as end-of-packet early.
    for k in 0..8u32 {
        let sz = (k as usize) * 255;
        let pkt = payload(0xA5A5 + k as u64, sz);
        let got = demux_all(&mux_packets(1, &[(&pkt, 42)]));
        assert_eq!(got.len(), 1, "k={k}");
        assert_eq!(got[0].data.len(), sz, "recovered length k={k}");
        assert_eq!(got[0].data, pkt, "recovered bytes k={k}");
    }
}

#[test]
fn many_small_packets_filling_and_overflowing_segment_tables() {
    // Enough 1-byte packets to overflow several segment tables (255/page). Ordering and
    // exact bytes must survive across the page boundaries.
    for count in [1usize, 100, 254, 255, 256, 510, 511, 700] {
        let pkts: Vec<Vec<u8>> = (0..count).map(|i| vec![(i as u8).wrapping_mul(37)]).collect();
        check_roundtrip(5, &pkts);
    }
}

#[test]
fn mixed_random_packet_sets() {
    // Property test: random mixes of sizes (biased toward small, with occasional huge
    // packets) round-trip exactly. Deterministic seed → reproducible.
    let mut r = Rng(0xDEAD_BEEF_CAFE);
    for trial in 0..40 {
        let n = 1 + r.below(60);
        let mut pkts = Vec::with_capacity(n);
        for _ in 0..n {
            let size = match r.below(10) {
                0 => 0,                      // nil packet
                1 => 255 * (1 + r.below(4)), // exact multiple of 255
                2 => 255,                    // exactly one full segment
                3..=4 => 200_000 + r.below(50_000), // huge, multi-page
                _ => r.below(400),           // the common small case
            };
            pkts.push(payload(trial * 131 + size as u64, size));
        }
        check_roundtrip(0x9000 + trial as u32, &pkts);
    }
}

#[test]
fn single_packet_spanning_many_pages() {
    // A packet far larger than one page (65025 payload bytes max/page) must split into
    // several continued pages and rejoin byte-exact.
    let pkt = payload(0x1234, 500_000);
    let got = demux_all(&mux_packets(77, &[(&pkt, 999)]));
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].data, pkt);
    assert!(got[0].bos && got[0].eos);
    assert_eq!(got[0].granule, 999, "granule lands on the finishing page");
}

#[test]
fn granule_positions_are_reported_per_finishing_page() {
    // One packet per page (force a flush between packets by using >1 page each): each
    // packet's granule should come back on its packet. Simpler: small packets share a
    // page, and the page's granule is the last packet to finish on it. Verify the
    // single-packet-per-page case with large packets.
    let pkts: Vec<Vec<u8>> = (0..4).map(|i| payload(i, 66_000)).collect(); // >1 page each
    let refs: Vec<(&[u8], u64)> = pkts
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_slice(), (i as u64 + 1) * 1000))
        .collect();
    let got = demux_all(&mux_packets(1, &refs));
    assert_eq!(got.len(), 4);
    for (i, p) in got.iter().enumerate() {
        assert_eq!(p.granule, (i as u64 + 1) * 1000, "granule of packet {i}");
    }
}

#[test]
fn empty_stream_just_eos() {
    // No packets written: finish() still emits a nil eos page — "a lacing value of
    // zero" (§5), a well-formed bos+eos page with an empty segment. The demuxer
    // faithfully surfaces that single zero-length packet (it cannot know the muxer
    // "meant nothing"); what matters is a valid stream and no panic.
    let mut w = OggWriter::new(1);
    let mut out = Vec::new();
    w.finish(&mut out);
    assert!(!out.is_empty(), "an eos page is emitted even with no packets");
    let got = demux_all(&out);
    assert_eq!(got.len(), 1, "the nil eos page reads back as one empty packet");
    assert!(got[0].data.is_empty(), "and that packet is zero-length");
    assert!(got[0].bos && got[0].eos, "it is the sole (bos+eos) page");
}

#[test]
fn streaming_reader_matches_oneshot_under_arbitrary_chunking() {
    // The push/pull reader fed in weird chunk sizes must produce identical packets to
    // demux_all over the whole buffer.
    let pkts: Vec<Vec<u8>> = vec![
        payload(1, 0),
        payload(2, 255),
        payload(3, 40),
        payload(4, 66_000),
        payload(5, 1),
        payload(6, 510),
    ];
    let refs: Vec<(&[u8], u64)> = pkts.iter().enumerate().map(|(i, p)| (p.as_slice(), i as u64)).collect();
    let stream = mux_packets(9, &refs);
    let oneshot = demux_all(&stream);

    for chunk in [1usize, 2, 3, 7, 13, 64, 997, 4096] {
        let mut reader = OggReader::new();
        let mut got: Vec<Packet> = Vec::new();
        for part in stream.chunks(chunk) {
            reader.push(part);
            while let Some(p) = reader.next_packet() {
                got.push(p);
            }
        }
        reader.finish();
        while let Some(p) = reader.next_packet() {
            got.push(p);
        }
        assert_eq!(got, oneshot, "chunk size {chunk} must match one-shot");
    }
}
