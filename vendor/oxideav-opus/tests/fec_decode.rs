//! In-band FEC (§4.2.5 LBRR) recovery validation (RFC 6716 §2.1.7).
//!
//! These tests exercise [`oxideav_opus::OpusDecoder::decode_packet_fec`],
//! which reconstructs a lost frame's audio from the §4.2.5 LBRR redundancy
//! carried in the *next* successfully received packet.
//!
//! ## Fixture
//!
//! `fec-on.opus` is a mono WB SILK Ogg-Opus stream encoded with in-band
//! FEC enabled (`-fec 1 -packet_loss 10`), so most packets carry an LBRR
//! copy of the prior 20 ms frame. It comes from the project's clean-room
//! Opus fixture corpus (a black-box encoder produced its bytes from a
//! 440 Hz synthetic source); see `docs/audio/opus/fixtures/fec-on/notes.md`.
//!
//! ## What is validated
//!
//! RFC 6716 §4.2.9 makes the SILK→48 kHz resampler non-normative, so a
//! bit-exact PCM comparison is not a meaningful conformance target (the
//! crate uses linear interpolation while the reference uses a polyphase
//! kernel). These tests therefore validate the FEC machinery
//! structurally:
//!
//! 1. **LBRR presence** — at least one packet of the FEC fixture reports
//!    [`FecDecodeStatus::Recovered`] (the encoder did emit LBRR data).
//! 2. **Sample-count accounting** — a recovered frame yields the carrier
//!    packet's per-frame 48 kHz sample count.
//! 3. **Signal content** — the recovered 440 Hz fixture audio is
//!    non-silent (a real reconstruction, not the silence floor).
//! 4. **No-FEC fallback** — a CELT-only / LBRR-absent packet reports a
//!    non-`Recovered` status and silence, never an error or a panic.
//!
//! No external library source is consulted.

use oxideav_opus::{FecDecodeStatus, OpusDecoder};

const FIXTURE_FEC: &[u8] = include_bytes!("fixtures/fec-on.opus");
const FIXTURE_NB_MONO: &[u8] = include_bytes!("fixtures/silk-nb-mono-16kbps.opus");
const FIXTURE_WB_STEREO: &[u8] = include_bytes!("fixtures/silk-wb-stereo-20kbps.opus");

/// Recover the raw Opus packets from an Ogg-Opus byte stream (the same
/// minimal, test-only page de-laker used by `silk_fixture_decode.rs`).
/// Drops the two RFC 7845 header packets (`OpusHead` / `OpusTags`).
fn ogg_audio_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut off = 0usize;
    let mut packets: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    while off + 27 <= data.len() {
        assert_eq!(&data[off..off + 4], b"OggS", "lost Ogg page sync at {off}");
        let nseg = data[off + 26] as usize;
        let seg_table_end = off + 27 + nseg;
        assert!(seg_table_end <= data.len(), "truncated Ogg lacing table");
        let segtab = &data[off + 27..seg_table_end];
        let mut p = seg_table_end;
        for &lace in segtab {
            let l = lace as usize;
            assert!(p + l <= data.len(), "Ogg segment overruns buffer");
            cur.extend_from_slice(&data[p..p + l]);
            p += l;
            if lace < 255 {
                packets.push(std::mem::take(&mut cur));
            }
        }
        off = p;
    }
    // RFC 7845: the first two packets are OpusHead + OpusTags.
    packets.into_iter().skip(2).collect()
}

#[test]
fn fec_fixture_recovers_at_least_one_frame() {
    let packets = ogg_audio_packets(FIXTURE_FEC);
    assert!(!packets.is_empty(), "no audio packets parsed from fec-on");

    let mut dec = OpusDecoder::new();
    let mut recovered = 0usize;
    let mut recovered_nonsilent = 0usize;

    for pkt in &packets {
        let fec = dec
            .decode_packet_fec(pkt)
            .expect("fec decode must not error");
        // The carrier is mono WB SILK 20 ms: 960 samples/frame at 48 kHz.
        assert_eq!(fec.channels, 1);
        assert_eq!(fec.sample_rate_hz, 48_000);
        assert_eq!(fec.pcm.len(), 960, "recovered frame is one 20 ms frame");

        match fec.status {
            FecDecodeStatus::Recovered => {
                recovered += 1;
                if fec.pcm.iter().any(|&s| s != 0) {
                    recovered_nonsilent += 1;
                }
            }
            FecDecodeStatus::NoLbrr => {}
            other => panic!("unexpected FEC status on a SILK packet: {other:?}"),
        }
    }

    assert!(
        recovered > 0,
        "the FEC-enabled fixture must recover at least one frame from LBRR"
    );
    assert!(
        recovered_nonsilent > 0,
        "a recovered 440 Hz frame must be non-silent (real reconstruction)"
    );
}

#[test]
fn fec_decode_advances_then_regular_decode_continues() {
    // Simulate a single packet loss: drop the regular decode of packet N,
    // recover it via FEC from packet N (carrier), then continue the
    // regular stream on packet N. The decoder must not error and the
    // subsequent regular decode must still produce a full-length frame.
    let packets = ogg_audio_packets(FIXTURE_FEC);
    assert!(packets.len() >= 3);

    let mut dec = OpusDecoder::new();
    // Warm up on the first packet (no prior loss).
    dec.decode_packet(&packets[0]).expect("warmup decode");

    // Pretend packet[1]'s audio was lost; recover from packet[2]'s LBRR.
    let fec = dec.decode_packet_fec(&packets[2]).expect("fec decode");
    assert!(matches!(
        fec.status,
        FecDecodeStatus::Recovered | FecDecodeStatus::NoLbrr
    ));

    // Now decode packet[2] normally; it must still yield a clean frame.
    let out = dec.decode_packet(&packets[2]).expect("regular decode");
    assert_eq!(out.channels, 1);
    assert_eq!(out.samples_per_channel(), 960);
}

#[test]
fn fec_on_non_fec_stream_reports_no_lbrr_or_recovered() {
    // The plain (no-FEC) NB mono fixture should not carry LBRR; every
    // packet must report a non-error status and silence (or, if the
    // encoder happened to emit any LBRR, a clean recovery) — never a
    // decode error or panic.
    let packets = ogg_audio_packets(FIXTURE_NB_MONO);
    let mut dec = OpusDecoder::new();
    for pkt in &packets {
        let fec = dec
            .decode_packet_fec(pkt)
            .expect("fec decode must not error");
        assert_eq!(fec.channels, 1);
        assert_eq!(fec.pcm.len(), 960);
        assert!(
            matches!(
                fec.status,
                FecDecodeStatus::NoLbrr | FecDecodeStatus::Recovered
            ),
            "non-FEC stream must not yield a decode error: {:?}",
            fec.status
        );
        if matches!(fec.status, FecDecodeStatus::NoLbrr) {
            assert!(fec.pcm.iter().all(|&s| s == 0), "NoLbrr must emit silence");
        }
    }
}

#[test]
fn fec_on_stereo_stream_routes_cleanly() {
    // The WB stereo fixture has no FEC, but the stereo FEC routing
    // (`decode_silk_fec_stereo`) must run without panicking and report a
    // clean status with the correct 2-channel, full-length silence on the
    // NoLbrr path.
    let packets = ogg_audio_packets(FIXTURE_WB_STEREO);
    let mut dec = OpusDecoder::new();
    for pkt in &packets {
        let fec = dec
            .decode_packet_fec(pkt)
            .expect("stereo fec decode must not error");
        assert_eq!(fec.channels, 2);
        assert_eq!(fec.sample_rate_hz, 48_000);
        // Stereo 20 ms = 960 per channel, interleaved => 1920 samples.
        assert_eq!(fec.pcm.len(), 1920);
        assert!(
            matches!(
                fec.status,
                FecDecodeStatus::NoLbrr | FecDecodeStatus::Recovered
            ),
            "stereo SILK packet must route cleanly: {:?}",
            fec.status
        );
        if matches!(fec.status, FecDecodeStatus::NoLbrr) {
            assert!(
                fec.pcm.iter().all(|&s| s == 0),
                "NoLbrr stereo must emit interleaved silence"
            );
        }
    }
}
