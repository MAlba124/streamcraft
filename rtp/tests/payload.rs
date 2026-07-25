//! Payload-format integration tests: H.264 (RFC 6184) and Opus (RFC 7587)
//! pay ↔ depay round trips, malformed-input loudness, and ffmpeg interop
//! fixtures.
//!
//! # Fixture format: `.rtpdump`
//!
//! Our own trivial container (NOT the rtpdump(1) format): a flat sequence of
//! records, each a **u32 little-endian byte length followed by one UDP
//! datagram, verbatim**. Datagrams appear in reception order.
//!
//! # Fixture generation (done ONCE on the dev machine, files checked in)
//!
//! Recorder — a throwaway std-only binary (`rustc -O rtprec.rs`) that binds
//! `127.0.0.1:<port>` and appends each received datagram to the dump:
//!
//! ```text
//! use std::{io::Write, net::UdpSocket, time::Duration};
//! fn main() {
//!     let mut a = std::env::args().skip(1);
//!     let sock = UdpSocket::bind(("127.0.0.1", a.next().unwrap().parse::<u16>().unwrap())).unwrap();
//!     sock.set_read_timeout(Some(Duration::from_secs(15))).unwrap();
//!     let mut out = std::io::BufWriter::new(std::fs::File::create(a.next().unwrap()).unwrap());
//!     let mut buf = [0u8; 65536];
//!     while let Ok(n) = sock.recv(&mut buf) {
//!         out.write_all(&(n as u32).to_le_bytes()).unwrap();
//!         out.write_all(&buf[..n]).unwrap();
//!         sock.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
//!     }
//!     out.flush().unwrap();
//! }
//! ```
//!
//! H.264 (`h264_testsrc.rtpdump` + `h264_testsrc.h264`), recorder on 5004:
//!
//! ```text
//! ffmpeg -re -f lavfi -i testsrc2=duration=2:size=320x240:rate=25 \
//!   -c:v libx264 -preset ultrafast -crf 30 -x264-params threads=1 \
//!   -f tee -map 0:v "[f=rtp]rtp://127.0.0.1:5004|[f=h264]rtp/tests/fixtures/h264_testsrc.h264"
//! ```
//!
//! Deviations from the plain `-f rtp` baseline, and why:
//! - `-f tee "[f=rtp]…|[f=h264]…"`: fans ONE libx264 encode into both the
//!   RTP output and the raw Annex-B sibling, so the `.h264` file is the very
//!   elementary stream that was packetized — the determinism needed for the
//!   byte-identity assertion (no second encode to drift).
//! - `-re`: paces packets at the media rate; an unpaced blast on loopback
//!   can overflow the recorder's socket buffer and silently drop datagrams
//!   (the test asserts the capture is gap-free).
//! - `-crf 30`: keeps each fixture under the ~150 KB budget (defaults gave
//!   ~175 KB). `-x264-params threads=1` removes thread-count nondeterminism.
//!
//! Opus (`opus_sine.rtpdump`), recorder on 5006:
//!
//! ```text
//! ffmpeg -re -f lavfi -i sine=duration=1 -c:a libopus -f rtp rtp://127.0.0.1:5006
//! ```
//!
//! Capture observations pinned by the tests: H.264 goes out as PT 96 with
//! SPS/PPS/SEI aggregated in a leading STAP-A and every slice FU-A'd (50
//! frames / 103 datagrams); Opus goes out as PT 97, one 20 ms CELT-FB packet
//! per datagram (51 datagrams). ffmpeg sets the RTP marker on *every* Opus
//! packet — RFC 3551 §4.1 wants it zero without silence suppression — so the
//! Opus test deliberately does not assert marker semantics.

use sc_rtp::depay::h264::{H264Depay, H264DepayError};
use sc_rtp::depay::opus::depay as opus_depay;
use sc_rtp::packet::RtpPacket;
use sc_rtp::pay::h264::pay as h264_pay;
use sc_rtp::pay::opus::pay as opus_pay;

// ---------------------------------------------------------------- helpers

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Read a `.rtpdump` fixture (u32 LE length + datagram, repeated).
fn rtpdump(name: &str) -> Vec<Vec<u8>> {
    let path = fixture_path(name);
    let d = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < d.len() {
        assert!(at + 4 <= d.len(), "truncated record length in {name}");
        let len = u32::from_le_bytes(d[at..at + 4].try_into().unwrap()) as usize;
        at += 4;
        assert!(at + len <= d.len(), "truncated datagram in {name}");
        out.push(d[at..at + len].to_vec());
        at += len;
    }
    out
}

/// Split an Annex-B stream into NAL units (3- and 4-byte start codes; one
/// zero before `00 00 01` belongs to the 4-byte form). Test-local twin of
/// the payloader's splitter so the two are pinned against each other.
fn split_nals(stream: &[u8]) -> Vec<&[u8]> {
    let mut positions = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i..i + 3] == [0, 0, 1] {
            positions.push(i);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut nals = Vec::new();
    for (k, &p) in positions.iter().enumerate() {
        let start = p + 3;
        let mut end = positions.get(k + 1).copied().unwrap_or(stream.len());
        if k + 1 < positions.len() && end > start && stream[end - 1] == 0 {
            end -= 1;
        }
        if end > start {
            nals.push(&stream[start..end]);
        }
    }
    nals
}

/// Deterministic xorshift32 for synthetic NAL bodies.
struct Rng(u32);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// Random NAL body without 0x00 bytes: real NAL payloads are
    /// emulation-prevented so `00 00 0x` never appears; the tests get the
    /// same guarantee by avoiding zeros altogether, keeping the Annex-B
    /// round trip bit-exact.
    fn body(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next_u32() % 255) as u8 + 1).collect()
    }
}

/// One NAL unit: `F=0 | NRI | type` header octet plus body.
fn nal(nri: u8, ty: u8, body: Vec<u8>) -> Vec<u8> {
    let mut v = vec![(nri << 5) | ty];
    v.extend(body);
    v
}

/// Annex-B framing with 4-byte start codes — the depayloader's output form,
/// so round trips compare bit-exact.
fn annexb(nals: &[Vec<u8>]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}

/// Synthetic access units exercising every packetization shape at `mtu`:
/// STAP-A (parameter-set run), FU-A (NAL > MTU), exact-fit single NAL,
/// minimal fragmentation (MTU+1), multi-NAL AUs, and a tiny NAL.
fn synthetic_aus(mtu: usize, rng: &mut Rng) -> Vec<Vec<u8>> {
    vec![
        // Parameter sets + big IDR: STAP-A run then FU-A.
        annexb(&[
            nal(3, 7, rng.body(20)),
            nal(3, 8, rng.body(6)),
            nal(3, 5, rng.body(4000)),
        ]),
        // Exactly MTU-sized NAL: the single-NAL boundary case (§5.6).
        annexb(&[nal(2, 1, rng.body(mtu - 1))]),
        // One byte over: minimal FU-A (§5.8).
        annexb(&[nal(2, 1, rng.body(mtu))]),
        // Several NALs, mixed aggregation/fragmentation fates.
        annexb(&[
            nal(2, 1, rng.body(90)),
            nal(0, 1, rng.body(60)),
            nal(2, 1, rng.body(700)),
        ]),
        // Tiny AU.
        annexb(&[nal(2, 1, rng.body(1))]),
    ]
}

// ------------------------------------------------------ H.264 round trips

#[test]
fn h264_round_trip_bit_exact_across_mtus() {
    for &mtu in &[500usize, 1400] {
        let mut rng = Rng(0x5EED_0000 | mtu as u32);
        let aus = synthetic_aus(mtu, &mut rng);
        let mut depay = H264Depay::new(Vec::new());
        let mut got = Vec::new();
        for (k, au) in aus.iter().enumerate() {
            let ts = 90_000 + 3_600 * k as u32;
            let packets = h264_pay(au, mtu);
            assert!(!packets.is_empty());
            for (i, (p, m)) in packets.iter().enumerate() {
                assert!(p.len() <= mtu, "payload {} > mtu {mtu}", p.len());
                assert_eq!(*m, i == packets.len() - 1, "marker on the last packet only (§5.1)");
                got.extend(depay.push(p, *m, ts).expect("clean stream"));
            }
        }
        assert_eq!(got, aus, "mtu {mtu}: depay(pay(au)) must be bit-exact");
    }
}

#[test]
fn h264_no_markers_falls_back_to_timestamp_boundary() {
    // §5.1: receivers MUST NOT rely on the marker; strip it entirely and the
    // timestamp change must delimit AUs, with flush() draining the last one.
    let mtu = 500;
    let mut rng = Rng(0xB0B);
    let aus = synthetic_aus(mtu, &mut rng);
    let mut depay = H264Depay::new(Vec::new());
    let mut got = Vec::new();
    for (k, au) in aus.iter().enumerate() {
        let ts = 3_600 * k as u32;
        for (p, _m) in h264_pay(au, mtu) {
            got.extend(depay.push(&p, false, ts).expect("clean stream"));
        }
    }
    got.extend(depay.flush());
    assert_eq!(got, aus);
}

#[test]
fn h264_two_aus_can_complete_in_one_push() {
    // AU1 never got its marker (lost); AU2 is a single marked packet at the
    // next timestamp: one push must yield both.
    let au1 = annexb(&[nal(2, 1, Rng(1).body(50))]);
    let au2 = annexb(&[nal(2, 1, Rng(2).body(40))]);
    let mut depay = H264Depay::new(Vec::new());

    let p1 = h264_pay(&au1, 1400);
    assert_eq!(p1.len(), 1);
    assert!(depay.push(&p1[0].0, false, 1000).unwrap().is_empty());

    let p2 = h264_pay(&au2, 1400);
    let got = depay.push(&p2[0].0, true, 4600).unwrap();
    assert_eq!(got, vec![au1, au2]);
}

#[test]
fn h264_sprop_parameter_sets_prepend_only_the_first_au() {
    // §8.1: sprop NALs precede any other NAL unit in decoding order.
    let sps = nal(3, 7, Rng(3).body(15));
    let pps = nal(3, 8, Rng(4).body(5));
    let au1 = annexb(&[nal(2, 1, Rng(5).body(80))]);
    let au2 = annexb(&[nal(2, 1, Rng(6).body(70))]);

    let mut depay = H264Depay::new(vec![sps.clone(), pps.clone()]);
    let mut got = Vec::new();
    for (k, au) in [&au1, &au2].into_iter().enumerate() {
        for (p, m) in h264_pay(au, 1400) {
            got.extend(depay.push(&p, m, 7_000 * (k as u32 + 1)).unwrap());
        }
    }
    let mut want_first = annexb(&[sps, pps]);
    want_first.extend_from_slice(&au1);
    assert_eq!(got, vec![want_first, au2]);
}

#[test]
fn h264_pay_aggregates_parameter_sets_into_stap_a() {
    let sps = nal(3, 7, Rng(7).body(20));
    let pps = nal(3, 8, Rng(8).body(6));
    let idr = nal(3, 5, Rng(9).body(100));
    let au = annexb(&[sps.clone(), pps.clone(), idr.clone()]);

    let packets = h264_pay(&au, 1400);
    // Everything fits one STAP-A: 1 + Σ(2 + len) < 1400.
    assert_eq!(packets.len(), 1);
    let (p, marker) = &packets[0];
    assert!(marker);
    assert_eq!(p[0] & 0x1F, 24, "STAP-A packet type (§5.7.1)");
    assert_eq!((p[0] >> 5) & 0x03, 3, "STAP NRI is the max of the aggregated NALs (§5.7)");
    // Aggregation units: 16-bit size + NAL, in order (§5.7.1).
    let mut units = Vec::new();
    let mut rest = &p[1..];
    while !rest.is_empty() {
        let size = u16::from_be_bytes([rest[0], rest[1]]) as usize;
        units.push(rest[2..2 + size].to_vec());
        rest = &rest[2 + size..];
    }
    assert_eq!(units, vec![sps, pps, idr]);
}

#[test]
fn h264_pay_fragments_oversize_nals_as_fu_a() {
    let mtu = 500;
    let big = nal(2, 5, Rng(10).body(3000));
    let au = annexb(std::slice::from_ref(&big));
    let packets = h264_pay(&au, mtu);
    assert!(packets.len() >= 2);

    let mut reassembled = vec![big[0]];
    for (i, (p, m)) in packets.iter().enumerate() {
        assert!(p.len() <= mtu);
        assert_eq!(p[0] & 0x1F, 28, "FU-A indicator type (§5.8)");
        assert_eq!(p[0] & 0xE0, big[0] & 0xE0, "indicator carries the NAL's F+NRI (§5.8)");
        assert_eq!(p[1] & 0x1F, big[0] & 0x1F, "FU header carries the NAL's type (§5.8)");
        let (s, e) = (p[1] & 0x80 != 0, p[1] & 0x40 != 0);
        assert_eq!(s, i == 0, "S only on the first fragment (§5.8)");
        assert_eq!(e, i == packets.len() - 1, "E only on the last fragment (§5.8)");
        assert_eq!(*m, i == packets.len() - 1, "marker on the AU's last packet (§5.1)");
        reassembled.extend_from_slice(&p[2..]);
    }
    assert_eq!(reassembled, big);
}

// -------------------------------------------------- H.264 malformed input

#[test]
fn h264_interleaved_packet_types_fail_loudly() {
    // STAP-B (25), MTAP16 (26), MTAP24 (27), FU-B (29): mode 2 only (§5.4
    // Table 3) — out of scope, must be Unsupported, never silence.
    for ty in [25u8, 26, 27, 29] {
        let mut depay = H264Depay::new(Vec::new());
        let payload = [(3 << 5) | ty, 0x00, 0x10, 0x41, 0xAA];
        assert_eq!(
            depay.push(&payload, false, 1000),
            Err(H264DepayError::Unsupported { packet_type: ty }),
        );
    }
}

#[test]
fn h264_truncated_fu_a_fails() {
    let mut depay = H264Depay::new(Vec::new());
    // Just the FU indicator, no FU header (§5.8 needs two octets).
    assert_eq!(depay.push(&[0x7C], false, 0), Err(H264DepayError::Truncated));
    // Empty payload.
    assert_eq!(depay.push(&[], false, 0), Err(H264DepayError::Truncated));
}

#[test]
fn h264_fu_a_end_without_start_fails() {
    let mut depay = H264Depay::new(Vec::new());
    // E=1 continuation with no fragment open: the start was lost and no
    // discontinuity was declared.
    let payload = [0x7C, 0x45, 0xAA, 0xBB]; // FU-A, E=1, type 5
    assert_eq!(depay.push(&payload, true, 0), Err(H264DepayError::BadFragmentation));
}

#[test]
fn h264_fu_a_start_and_end_together_fails() {
    // §5.8: S and E MUST NOT both be set in one FU header.
    let mut depay = H264Depay::new(Vec::new());
    let payload = [0x7C, 0xC5, 0xAA];
    assert_eq!(depay.push(&payload, true, 0), Err(H264DepayError::BadFragmentation));
}

#[test]
fn h264_marker_mid_fragment_fails() {
    // Marker claims the AU ends here, but the fragmented NAL has no E yet.
    let mut depay = H264Depay::new(Vec::new());
    let start = [0x7C, 0x85, 0xAA, 0xBB]; // FU-A S=1 type 5
    assert_eq!(depay.push(&start, true, 0), Err(H264DepayError::BadFragmentation));
}

#[test]
fn h264_interrupted_fragment_fails() {
    let mut depay = H264Depay::new(Vec::new());
    let start = [0x7C, 0x85, 0xAA]; // FU-A S=1 type 5
    assert!(depay.push(&start, false, 0).unwrap().is_empty());
    // A single-NAL packet in the middle of the fragment run (§5.8: fragments
    // are consecutive).
    let single = [0x41, 0x01, 0x02];
    assert_eq!(depay.push(&single, false, 0), Err(H264DepayError::BadFragmentation));
}

#[test]
fn h264_malformed_stap_a_fails() {
    // Aggregation-unit size runs past the payload (§5.7.1).
    let mut depay = H264Depay::new(Vec::new());
    let truncated = [0x78, 0x00, 0x40, 0x41, 0xAA]; // claims 64 bytes, has 2
    assert_eq!(depay.push(&truncated, true, 0), Err(H264DepayError::Truncated));

    // Zero-size aggregation unit.
    let mut depay = H264Depay::new(Vec::new());
    let zero = [0x78, 0x00, 0x00];
    assert_eq!(depay.push(&zero, true, 0), Err(H264DepayError::BadAggregation));

    // Nested aggregation (§5.7: MUST NOT be nested).
    let mut depay = H264Depay::new(Vec::new());
    let nested = [0x78, 0x00, 0x02, 0x78, 0x01];
    assert_eq!(depay.push(&nested, true, 0), Err(H264DepayError::BadAggregation));

    // Empty STAP-A (§5.7.1: at least one aggregation unit).
    let mut depay = H264Depay::new(Vec::new());
    assert_eq!(depay.push(&[0x78], true, 0), Err(H264DepayError::BadAggregation));
}

#[test]
fn h264_reserved_types_are_ignored_not_fatal() {
    // §5.4 Table 3: packet types 0 and 30-31 "MUST be ignored by a receiver".
    let slice = nal(2, 1, Rng(11).body(30));
    let au = annexb(std::slice::from_ref(&slice));
    let mut depay = H264Depay::new(Vec::new());
    assert!(depay.push(&[0x00, 0xAA], false, 500).unwrap().is_empty());
    assert!(depay.push(&[0x1E, 0xAA], false, 500).unwrap().is_empty());
    assert!(depay.push(&[0x1F, 0xAA], false, 500).unwrap().is_empty());
    // ...and they must not have disturbed AU assembly.
    let got = depay.push(&slice, true, 500).unwrap();
    assert_eq!(got, vec![au]);
}

#[test]
fn h264_error_drops_partial_au_and_recovers_at_next_timestamp() {
    let mut depay = H264Depay::new(Vec::new());
    // Open a fragment in AU@1000...
    assert!(depay.push(&[0x7C, 0x85, 0xAA], false, 1000).unwrap().is_empty());
    // ...then violate the FU state machine (second S=1 while open).
    assert_eq!(
        depay.push(&[0x7C, 0x85, 0xBB], false, 1000),
        Err(H264DepayError::BadFragmentation)
    );
    // Same-timestamp stragglers are skipped silently while resynchronizing.
    assert!(depay.push(&[0x7C, 0x05, 0xCC], false, 1000).unwrap().is_empty());
    // The next timestamp provably starts a fresh AU — decoded in full, and
    // nothing of the broken AU@1000 leaks into it.
    let slice = nal(2, 1, Rng(12).body(25));
    let got = depay.push(&slice, true, 4600).unwrap();
    assert_eq!(got, vec![annexb(&[slice])]);
}

#[test]
fn h264_discontinuity_drops_partial_au_and_resyncs() {
    let head = nal(2, 1, Rng(13).body(40));
    let tail_a = nal(2, 1, Rng(14).body(35));
    let tail_b = nal(2, 1, Rng(15).body(30));
    let fresh = nal(2, 5, Rng(16).body(45));

    let mut depay = H264Depay::new(Vec::new());
    // Partial AU@1000, then declared loss.
    assert!(depay.push(&head, false, 1000).unwrap().is_empty());
    depay.discontinuity();
    // Packets of AU@2000 may be the tail of an AU whose head was lost:
    // skipped, marker or not.
    assert!(depay.push(&tail_a, false, 2000).unwrap().is_empty());
    assert!(depay.push(&tail_b, true, 2000).unwrap().is_empty());
    // The packet after the marker starts a fresh AU: delivered, without any
    // NAL from the dropped or skipped packets.
    let got = depay.push(&fresh, true, 3000).unwrap();
    assert_eq!(got, vec![annexb(&[fresh])]);
}

// --------------------------------------------------------- ffmpeg interop

#[test]
fn h264_ffmpeg_fixture_depayloads_to_the_encoders_stream() {
    let dgrams = rtpdump("h264_testsrc.rtpdump");
    assert!(!dgrams.is_empty());

    let mut depay = H264Depay::new(Vec::new());
    let mut aus: Vec<Vec<u8>> = Vec::new();
    let mut prev_seq: Option<u16> = None;
    for d in &dgrams {
        let p = RtpPacket::parse(d).expect("fixture datagrams are valid RTP");
        assert_eq!(p.payload_type(), 96, "ffmpeg's dynamic PT for H.264");
        if let Some(s) = prev_seq {
            assert_eq!(p.seq(), s.wrapping_add(1), "capture must be loss-free and in order");
        }
        prev_seq = Some(p.seq());
        aus.extend(
            depay
                .push(p.payload(), p.marker(), p.timestamp())
                .expect("ffmpeg sends non-interleaved mode packets"),
        );
    }
    aus.extend(depay.flush());

    // testsrc2 duration=2 rate=25 → 50 coded frames → 50 access units.
    assert_eq!(aus.len(), 50);

    // Structural invariants: SPS/PPS/IDR present in-band (ffmpeg aggregates
    // them into the leading STAP-A), and only plain NAL types survive.
    let stream: Vec<u8> = aus.concat();
    let nals = split_nals(&stream);
    let types: Vec<u8> = nals.iter().map(|n| n[0] & 0x1F).collect();
    assert!(types.contains(&7), "SPS missing");
    assert!(types.contains(&8), "PPS missing");
    assert!(types.contains(&5), "IDR slice missing");
    assert!(types.iter().all(|t| (1..=23).contains(t)), "non-NAL type leaked: {types:?}");

    // Byte identity against the raw Annex-B sibling. The tee muxer fanned a
    // single libx264 encode into both fixture files, so the NAL unit
    // sequences must match byte for byte. Identity is asserted per NAL, not
    // on the whole byte stream: RFC 6184 transports NAL units, not their
    // Annex-B framing (§5.8/§7.1 reconstruct the unit), and the encoder's
    // own file mixes 3- and 4-byte start codes (its SEI uses `00 00 01`)
    // while the depayloader emits uniform 4-byte codes.
    let es = std::fs::read(fixture_path("h264_testsrc.h264")).unwrap();
    let es_nals = split_nals(&es);
    assert_eq!(nals.len(), es_nals.len(), "NAL count differs from the encoder's stream");
    for (i, (ours, theirs)) in nals.iter().zip(&es_nals).enumerate() {
        assert_eq!(ours, theirs, "NAL #{i} differs from the encoder's stream");
    }
}

/// RFC 6716 §3.1 (the framing RFC 7587 §4.2 defers to): TOC config → frame
/// duration in 48 kHz ticks, code → frame count.
fn opus_packet_duration_48k(pkt: &[u8]) -> u32 {
    let toc = pkt[0];
    let config = toc >> 3;
    let per_frame = match config {
        // SILK NB/MB/WB: 10/20/40/60 ms per 4-config band.
        0..=11 => [480u32, 960, 1920, 2880][(config % 4) as usize],
        // Hybrid SWB/FB: 10/20 ms.
        12..=15 => [480, 960][(config % 2) as usize],
        // CELT NB/WB/SWB/FB: 2.5/5/10/20 ms per 4-config band.
        _ => [120, 240, 480, 960][((config - 16) % 4) as usize],
    };
    let frames = match toc & 0x03 {
        0 => 1,
        1 | 2 => 2,
        _ => u32::from(pkt.get(1).expect("code-3 packet has a count byte") & 0x3F),
    };
    per_frame * frames
}

#[test]
fn opus_ffmpeg_fixture_depayloads_verbatim() {
    let dgrams = rtpdump("opus_sine.rtpdump");
    // Pinned at capture time: 1 s of 20 ms packets (+1 for encoder padding).
    assert_eq!(dgrams.len(), 51);

    let mut prev: Option<(u16, u32, u32)> = None; // seq, ts, duration
    let mut first_toc: Option<u8> = None;
    let mut total = 0u32;
    for d in &dgrams {
        let p = RtpPacket::parse(d).expect("fixture datagrams are valid RTP");
        assert_eq!(p.payload_type(), 97, "ffmpeg's dynamic PT for Opus");

        // §4.2: the RTP payload IS the Opus packet, byte for byte.
        let pkt = opus_depay(p.payload());
        assert_eq!(pkt, p.payload());
        assert!(!pkt.is_empty(), "an Opus packet has at least its TOC byte");

        // TOC sanity: one encoder, one mode — the config must not wander.
        assert_eq!(*first_toc.get_or_insert(pkt[0]), pkt[0], "TOC changed mid-stream");

        let dur = opus_packet_duration_48k(&pkt);
        assert!(dur > 0 && dur <= 120 * 48, "§4.2: at most 120 ms per packet");
        if let Some((seq, ts, prev_dur)) = prev {
            assert_eq!(p.seq(), seq.wrapping_add(1), "capture must be loss-free (and DTX-free)");
            // §4.1: the 48 kHz timestamp advances by the previous packet's
            // frame duration(s) (§4.2 Table 2).
            assert_eq!(p.timestamp().wrapping_sub(ts), prev_dur);
        }
        prev = Some((p.seq(), p.timestamp(), dur));
        total += dur;
    }
    assert!(total >= 48_000, "at least the full second of audio arrived");
}

#[test]
fn opus_round_trip_is_verbatim() {
    let mut rng = Rng(0x0905);
    for len in [1usize, 2, 3, 100, 1275] {
        let mut pkt = rng.body(len);
        pkt[0] = 0xF8; // any TOC; the payload format never inspects it
        assert_eq!(opus_depay(&opus_pay(&pkt)), pkt);
    }
}
