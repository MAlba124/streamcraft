//! LIVE end-to-end test against a local mediamtx RTSP server with an ffmpeg
//! publisher, plus an offline parse of the captured DESCRIBE fixture.
//!
//! The live test *requires a running server* and skips cleanly (print +
//! return, like sc-vaapi's hardware tests) when 127.0.0.1:8554 is
//! unreachable — or reachable but without a publisher on `/live` — so the
//! suite stays green on machines without the setup. To run it for real:
//!
//! ```text
//! nix run nixpkgs#mediamtx &          # RTSP server on :8554
//! ffmpeg -re -f lavfi -i testsrc2=duration=30:size=320x240:rate=25 \
//!     -c:v libx264 -preset ultrafast -f rtsp rtsp://127.0.0.1:8554/live &
//! cargo test -p sc-rtsp --test live_mediamtx
//! ```
//!
//! When live, the DESCRIBE body is (re)captured into
//! `tests/fixtures/mediamtx_describe.sdp` — the checked-in copy the offline
//! test parses, so real-server SDP shape is pinned even where the live test
//! skips.

use std::net::{SocketAddr, TcpStream, UdpSocket};
use std::time::{Duration, Instant};

use sc_rtsp::client::{resolve_control, RtspClient};
use sc_rtsp::sdp::MediaKind;

const SERVER: &str = "127.0.0.1:8554";
const URL: &str = "rtsp://127.0.0.1:8554/live";
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mediamtx_describe.sdp"
);

/// Bind an even/odd UDP port pair for RTP/RTCP (§12.39 client_port is a
/// range; RTP rides the even port by RTP convention, RFC 3550 §11).
fn bind_rtp_pair() -> Option<(UdpSocket, UdpSocket, u16)> {
    for base in (40000..40200u16).step_by(2) {
        if let (Ok(rtp), Ok(rtcp)) = (
            UdpSocket::bind(("127.0.0.1", base)),
            UdpSocket::bind(("127.0.0.1", base + 1)),
        ) {
            return Some((rtp, rtcp, base));
        }
    }
    None
}

#[test]
fn live_describe_setup_play_packets_teardown() {
    // Gate: server reachable? Skip cleanly otherwise.
    let addr: SocketAddr = SERVER.parse().unwrap();
    if TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_err() {
        eprintln!("skip live_describe_setup_play_packets_teardown: no RTSP server on {SERVER}");
        return;
    }

    let mut client = RtspClient::connect(URL).unwrap();

    let methods = client.options().unwrap();
    assert!(
        methods.iter().any(|m| m == "DESCRIBE"),
        "server Public should advertise DESCRIBE, got {methods:?}"
    );

    // A reachable server without a publisher 404s the DESCRIBE — still a
    // skip (the gate is about *this* machine's setup, not the protocol).
    let described = match client.describe() {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "skip live_describe_setup_play_packets_teardown: DESCRIBE failed \
                 (no publisher on {URL}?): {e}"
            );
            return;
        }
    };

    // Refresh the checked-in fixture whenever we are live. Write-then-rename
    // so the offline parse test (running in parallel in this same binary)
    // never observes a truncated file.
    let tmp = format!("{FIXTURE}.tmp");
    std::fs::write(&tmp, &described.raw).unwrap();
    std::fs::rename(&tmp, FIXTURE).unwrap();

    // The ffmpeg publisher sends H.264 video: find that media section.
    let video = described
        .sdp
        .media
        .iter()
        .find(|m| m.kind == MediaKind::Video)
        .expect("publisher SDP has a video section");
    assert!(
        video
            .rtpmap
            .values()
            .any(|r| r.encoding.eq_ignore_ascii_case("H264")),
        "expected an H264 rtpmap, got {:?}",
        video.rtpmap
    );

    let (rtp, _rtcp, base_port) = bind_rtp_pair().expect("no free UDP port pair");
    rtp.set_read_timeout(Some(Duration::from_millis(200)))
        .unwrap();

    let control = resolve_control(&described.base, video.control.as_deref().unwrap_or("*"));
    let transport = client.setup(&control, base_port).unwrap();
    assert!(
        transport.server_port.is_some(),
        "mediamtx returns server_port (§12.39), got {:?}",
        transport.raw
    );
    let session = client
        .session()
        .expect("SETUP establishes a session")
        .clone();
    assert!(!session.id.is_empty());

    client.play().unwrap();

    // RTP datagrams must arrive on our client_port within 2 s.
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut packets = 0usize;
    let mut buf = [0u8; 2048];
    while Instant::now() < deadline {
        // Err is a timeout tick — keep polling until the deadline.
        if let Ok(n) = rtp.recv(&mut buf) {
            // RTP fixed header is 12 bytes, version 2 in the top bits
            // (RFC 3550 §5.1) — sanity, not full validation.
            assert!(n >= 12, "datagram shorter than an RTP header");
            assert_eq!(buf[0] >> 6, 2, "not RTP version 2");
            packets += 1;
        }
    }
    assert!(
        packets >= 5,
        "expected RTP packets within 2s of PLAY, got {packets}"
    );

    // Keepalive rides the established session, then tear down.
    client.keepalive().unwrap();
    client.teardown().unwrap();
    assert!(client.session().is_none());

    eprintln!("live test: {packets} RTP packets in 2s, session torn down");
}

/// Offline: the captured mediamtx DESCRIBE body parses and looks like the
/// ffmpeg H.264 publish that produced it.
#[test]
fn fixture_mediamtx_describe_parses() {
    let raw = match std::fs::read_to_string(FIXTURE) {
        Ok(r) => r,
        Err(e) => panic!("fixture {FIXTURE} unreadable ({e}) — run the live test to capture it"),
    };
    let sdp = sc_rtsp::sdp::parse(&raw).unwrap();

    let video = sdp
        .media
        .iter()
        .find(|m| m.kind == MediaKind::Video)
        .expect("captured SDP has a video section");
    assert_eq!(video.proto, "RTP/AVP");
    assert!(!video.payload_types.is_empty());
    let (&pt, rtpmap) = video
        .rtpmap
        .iter()
        .find(|(_, r)| r.encoding.eq_ignore_ascii_case("H264"))
        .expect("H264 rtpmap");
    assert_eq!(
        rtpmap.clock_rate, 90000,
        "RFC 6184 §8.1: H264 clock is 90 kHz"
    );
    assert!(video.payload_types.contains(&pt));
    // Every RTSP-served media has a control attribute to SETUP against.
    assert!(video.control.is_some());

    // mediamtx includes sprop-parameter-sets in the fmtp; the SPS NAL type
    // is 7 and PPS 8 (low 5 bits of the first byte) once base64-decoded.
    if let Some(fmtp) = video.fmtp.get(&pt) {
        if let Some(sprop) = fmtp.params.get("sprop-parameter-sets") {
            let mut sets = sprop.split(',');
            let sps = sc_rtsp::sdp::decode_base64(sets.next().unwrap()).unwrap();
            assert_eq!(sps[0] & 0x1F, 7, "first parameter set is an SPS");
            let pps = sc_rtsp::sdp::decode_base64(sets.next().unwrap()).unwrap();
            assert_eq!(pps[0] & 0x1F, 8, "second parameter set is a PPS");
        }
    }
}
