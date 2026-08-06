//! Server tests driven by the crate's own client over real loopback TCP:
//! [`pf_rtsp::client::RtspClient`] speaks to [`pf_rtsp::server::RtspServer`]
//! on 127.0.0.1 — the full RFC 2326 happy path (OPTIONS → DESCRIBE → SETUP
//! → PLAY → keepalive → PAUSE → TEARDOWN with [`ServerEvent`]s observed),
//! the error catalogue (461/455/454/501/400, §11), session timeout
//! (§12.37), connection-drop teardown, and sequential sessions.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

use pf_rtsp::client::{resolve_control, RtspClient, RtspError};
use pf_rtsp::sdp::MediaKind;
use pf_rtsp::server::{RtspServer, ServerConfig, ServerEvent};

/// The presentation DESCRIBE serves: one H.264 video track whose
/// `a=control` matches the server's configured control name.
const SDP_BODY: &str = "v=0\r\n\
    o=- 1 1 IN IP4 127.0.0.1\r\n\
    s=server-loopback\r\n\
    t=0 0\r\n\
    a=control:*\r\n\
    m=video 0 RTP/AVP 96\r\n\
    a=rtpmap:96 H264/90000\r\n\
    a=control:trackID=0\r\n";

const CONTROL: &str = "trackID=0";
const SERVER_PORT: (u16, u16) = (6000, 6001);

/// A running server (with `timeout_secs` overridden) plus its event tap.
fn start(timeout_secs: u64) -> (RtspServer, Receiver<ServerEvent>) {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut config = ServerConfig::new(SDP_BODY, CONTROL);
    config.server_port = SERVER_PORT;
    config.timeout_secs = timeout_secs;
    let server = RtspServer::bind("127.0.0.1:0".parse().unwrap(), config, tx).unwrap();
    (server, rx)
}

/// A client connected to `server`'s real socket, presentation URL
/// `rtsp://127.0.0.1:<port>/stream`.
fn client_for(server: &RtspServer) -> RtspClient {
    RtspClient::connect(&format!("rtsp://{}/stream", server.local_addr())).unwrap()
}

fn next_event(rx: &Receiver<ServerEvent>) -> ServerEvent {
    rx.recv_timeout(Duration::from_secs(5))
        .expect("expected a ServerEvent")
}

/// 1. The full happy path, with the SDP round-tripping through the
/// client's parser and every app-visible event checked.
#[test]
fn full_happy_path_with_events() {
    let (server, events) = start(60);
    let mut client = client_for(&server);

    // OPTIONS (§10.1): the advertised set covers what we're about to use.
    let methods = client.options().unwrap();
    for needed in ["DESCRIBE", "SETUP", "PLAY", "PAUSE", "TEARDOWN", "GET_PARAMETER"] {
        assert!(methods.iter().any(|m| m == needed), "Public missing {needed}: {methods:?}");
    }

    // DESCRIBE (§10.2): the configured SDP arrives verbatim and parses.
    let d = client.describe().unwrap();
    assert_eq!(d.raw, SDP_BODY, "SDP served byte-for-byte");
    assert_eq!(d.sdp.media.len(), 1);
    assert_eq!(d.sdp.media[0].kind, MediaKind::Video);
    assert_eq!(d.sdp.media[0].control.as_deref(), Some(CONTROL));
    // Content-Base (§12.11/§C.1.1): request URL plus the trailing slash.
    assert_eq!(d.base, format!("rtsp://{}/stream/", server.local_addr()));

    // SETUP (§10.4/§12.39): client_port honored, session id + timeout
    // issued (§12.37, §3.4).
    let control_url = resolve_control(&d.base, d.sdp.media[0].control.as_deref().unwrap());
    let t = client.setup(&control_url, 5000).unwrap();
    assert_eq!(t.client_port, Some((5000, 5001)), "client_port echoed");
    assert_eq!(t.server_port, Some(SERVER_PORT), "configured server_port announced");
    let session = client.session().expect("SETUP names a session").clone();
    assert!(session.id.len() >= 8, "§3.4 session id length, got {:?}", session.id);
    assert_eq!(session.timeout_secs, 60, "§12.37 timeout advertised");

    // PLAY (§10.5): the app hears where to send.
    client.play().unwrap();
    match next_event(&events) {
        ServerEvent::Play {
            session: sid,
            client_rtp,
            client_rtcp,
        } => {
            assert_eq!(sid, session.id);
            assert_eq!(client_rtp.ip().to_string(), "127.0.0.1", "peer IP of the connection");
            assert_eq!(client_rtp.port(), 5000);
            assert_eq!(client_rtcp.port(), 5001);
        }
        other => panic!("expected Play, got {other:?}"),
    }

    // GET_PARAMETER keepalive (§10.8): 200, no body, no event.
    client.keepalive().unwrap();

    // PAUSE (§10.6).
    client.pause().unwrap();
    assert_eq!(
        next_event(&events),
        ServerEvent::Pause { session: session.id.clone() }
    );

    // PLAY again resumes (A.2: Ready --PLAY--> Playing).
    client.play().unwrap();
    assert!(matches!(next_event(&events), ServerEvent::Play { .. }));

    // TEARDOWN (§10.7): terminal event, session forgotten on both sides.
    client.teardown().unwrap();
    assert_eq!(
        next_event(&events),
        ServerEvent::Teardown { session: session.id.clone() }
    );
    assert!(client.session().is_none());

    // Shutdown produces no stray events: the session already ended with
    // TEARDOWN, so closing the connection emits no second one.
    server.shutdown();
    match events.try_recv() {
        Err(TryRecvError::Empty | TryRecvError::Disconnected) => {}
        Ok(ev) => panic!("unexpected event after teardown: {ev:?}"),
    }
}

/// 2. The error catalogue, each answered on a connection that stays
/// usable afterwards.
#[test]
fn error_paths_answer_loud_and_correct() {
    let (server, events) = start(60);
    let mut client = client_for(&server);
    let d = client.describe().unwrap();
    let control_url = resolve_control(&d.base, CONTROL);

    // SETUP offering TCP interleaving → 461 Unsupported Transport
    // (§11.3.12; v1 is UDP-unicast only).
    let resp = client
        .request(
            "SETUP",
            &control_url,
            &[("Transport", "RTP/AVP/TCP;unicast;interleaved=0-1")],
            &[],
        )
        .unwrap();
    assert_eq!(resp.status, 461);

    // SETUP on the presentation (aggregate) URL → 459 (§11.3.10; v1 has
    // no aggregate multi-track SETUP).
    let resp = client
        .request(
            "SETUP",
            &d.base,
            &[("Transport", "RTP/AVP;unicast;client_port=5000-5001")],
            &[],
        )
        .unwrap();
    assert_eq!(resp.status, 459);

    // PLAY without any SETUP → 455 Method Not Valid In This State
    // (§11.3.6; A.2 lists no PLAY transition in Init).
    match client.play() {
        Err(RtspError::Status(455, _)) => {}
        other => panic!("expected 455, got {other:?}"),
    }

    // A bogus session id → 454 Session Not Found (§11.3.5).
    let base = d.base.clone();
    let resp = client
        .request("PLAY", &base, &[("Session", "deadbeefcafef00d")], &[])
        .unwrap();
    assert_eq!(resp.status, 454);

    // Unknown / unimplemented methods → 501 Not Implemented (§10 Table 2
    // note), for both an extension-method and a known-but-unserved one.
    let resp = client.request("FROBNICATE", &base, &[], &[]).unwrap();
    assert_eq!(resp.status, 501);
    let resp = client.request("RECORD", &base, &[], &[]).unwrap();
    assert_eq!(resp.status, 501);

    // A second DESCRIBE on the same connection is fine (§10.1: errors
    // above left no state behind).
    let d2 = client.describe().unwrap();
    assert_eq!(d2.raw, SDP_BODY);

    // SETUP without a Transport header → 400 (nothing offered to choose
    // from, §12.39).
    let resp = client.request("SETUP", &control_url, &[], &[]).unwrap();
    assert_eq!(resp.status, 400);

    // Now a real session: PAUSE while Ready (never played) → 455 (A.2's
    // Ready row has no PAUSE transition).
    client.setup(&control_url, 5010).unwrap();
    match client.pause() {
        Err(RtspError::Status(455, _)) => {}
        other => panic!("expected 455, got {other:?}"),
    }
    // ... and the session is still intact: TEARDOWN works and reaches the
    // app. (No Play/Pause events were ever emitted on the error paths.)
    let sid = client.session().unwrap().id.clone();
    client.teardown().unwrap();
    assert_eq!(next_event(&events), ServerEvent::Teardown { session: sid });

    server.shutdown();
    match events.try_recv() {
        Err(_) => {}
        Ok(ev) => panic!("error paths must not emit events, got {ev:?}"),
    }
}

/// Read exactly one RTSP response off a raw socket: head to the blank
/// line, then the `Content-Length` body if declared (§4.4) — so nothing
/// of one response bleeds into the next read.
fn read_raw_response(stream: &mut TcpStream) -> String {
    let mut buf: Vec<u8> = Vec::new();
    loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let head_len = pos + 4;
            let head = String::from_utf8_lossy(&buf[..head_len]).into_owned();
            let body_len = head
                .lines()
                .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(str::to_owned))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if buf.len() >= head_len + body_len {
                return String::from_utf8_lossy(&buf[..head_len + body_len]).into_owned();
            }
        }
        let mut chunk = [0u8; 1024];
        let n = stream.read(&mut chunk).expect("read response");
        assert!(n > 0, "connection closed mid-response; got {buf:?}");
        buf.extend_from_slice(&chunk[..n]);
    }
}

/// 2b. Malformed requests → 400 (§7.1.1), exercised on a raw socket since
/// the client cannot be convinced to send garbage.
#[test]
fn malformed_requests_get_400() {
    // An unparsable request line: 400 and the connection is dropped
    // (framing can no longer be trusted).
    let (server, _events) = start(60);
    let mut raw = TcpStream::connect(server.local_addr()).unwrap();
    raw.write_all(b"NOT A VALID REQUEST\r\n\r\n").unwrap();
    let mut buf = Vec::new();
    raw.read_to_end(&mut buf).unwrap(); // server closes after answering
    let text = String::from_utf8_lossy(&buf);
    assert!(text.starts_with("RTSP/1.0 400 "), "got {text:?}");

    // A parseable request missing CSeq: 400 (§12.17: "This field MUST be
    // present in all requests"), but the connection survives and a
    // well-formed request still works.
    let mut raw = TcpStream::connect(server.local_addr()).unwrap();
    raw.write_all(b"OPTIONS rtsp://x/ RTSP/1.0\r\n\r\n").unwrap();
    let text = read_raw_response(&mut raw);
    assert!(text.starts_with("RTSP/1.0 400 "), "got {text:?}");
    raw.write_all(b"OPTIONS rtsp://x/ RTSP/1.0\r\nCSeq: 7\r\n\r\n")
        .unwrap();
    let text = read_raw_response(&mut raw);
    assert!(text.starts_with("RTSP/1.0 200 OK"), "got {text:?}");
    assert!(text.contains("CSeq: 7"), "§12.17 echo, got {text:?}");

    // A foreign RTSP version → 505 (§7.1.1), connection kept.
    raw.write_all(b"OPTIONS rtsp://x/ RTSP/2.0\r\nCSeq: 8\r\n\r\n")
        .unwrap();
    let text = read_raw_response(&mut raw);
    assert!(text.starts_with("RTSP/1.0 505 "), "got {text:?}");

    server.shutdown();
}

/// 3. Session timeout (§12.37/A.2): SETUP with a short timeout, send no
/// keepalive, and the app receives the Teardown.
#[test]
fn session_times_out_without_keepalive() {
    let (server, events) = start(1);
    let mut client = client_for(&server);
    let d = client.describe().unwrap();
    let control_url = resolve_control(&d.base, CONTROL);
    client.setup(&control_url, 5020).unwrap();
    let session = client.session().unwrap().clone();
    assert_eq!(session.timeout_secs, 1, "§12.37: configured timeout advertised");

    // No keepalive: within a few poll ticks past the 1 s budget the
    // server reverts to Init and tells the app.
    assert_eq!(
        next_event(&events),
        ServerEvent::Teardown { session: session.id.clone() }
    );

    // The stale id is now 454 territory (§11.3.5: "has timed out") — the
    // client still auto-sends it.
    match client.play() {
        Err(RtspError::Status(454, _)) => {}
        other => panic!("expected 454 after timeout, got {other:?}"),
    }

    // Teardown is exactly-once: no second event for the same session.
    server.shutdown();
    match events.try_recv() {
        Err(_) => {}
        Ok(ev) => panic!("timeout teardown must be exactly-once, got {ev:?}"),
    }
}

/// 3b. Keepalives actually hold the session open past its timeout.
#[test]
fn keepalive_refreshes_the_session_timer() {
    let (server, events) = start(1);
    let mut client = client_for(&server);
    let d = client.describe().unwrap();
    let control_url = resolve_control(&d.base, CONTROL);
    client.setup(&control_url, 5030).unwrap();
    let session = client.session().unwrap().clone();

    // Ping every 400 ms for 2 s — well past the 1 s timeout.
    for _ in 0..5 {
        std::thread::sleep(Duration::from_millis(400));
        client.keepalive().unwrap();
    }
    assert!(
        matches!(events.try_recv(), Err(TryRecvError::Empty)),
        "keepalives must prevent the timeout teardown"
    );

    client.teardown().unwrap();
    assert_eq!(next_event(&events), ServerEvent::Teardown { session: session.id });
    server.shutdown();
}

/// 4. Two sequential clients: teardown, then a fresh session works and
/// gets a distinct id.
#[test]
fn sequential_clients_get_fresh_sessions() {
    let (server, events) = start(60);

    let mut c1 = client_for(&server);
    let d = c1.describe().unwrap();
    let control_url = resolve_control(&d.base, CONTROL);
    c1.setup(&control_url, 5040).unwrap();
    let s1 = c1.session().unwrap().id.clone();
    c1.play().unwrap();
    assert!(matches!(next_event(&events), ServerEvent::Play { session, .. } if session == s1));
    c1.teardown().unwrap();
    assert_eq!(next_event(&events), ServerEvent::Teardown { session: s1.clone() });
    drop(c1);

    let mut c2 = client_for(&server);
    let d = c2.describe().unwrap();
    let control_url = resolve_control(&d.base, CONTROL);
    let t = c2.setup(&control_url, 5042).unwrap();
    assert_eq!(t.client_port, Some((5042, 5043)));
    let s2 = c2.session().unwrap().id.clone();
    assert_ne!(s1, s2, "§3.4: fresh session, fresh id");
    c2.play().unwrap();
    match next_event(&events) {
        ServerEvent::Play { session, client_rtp, client_rtcp } => {
            assert_eq!(session, s2);
            assert_eq!(client_rtp.port(), 5042);
            assert_eq!(client_rtcp.port(), 5043);
        }
        other => panic!("expected Play, got {other:?}"),
    }
    c2.teardown().unwrap();
    assert_eq!(next_event(&events), ServerEvent::Teardown { session: s2 });

    server.shutdown();
}

/// A client that vanishes mid-session (no TEARDOWN) still yields the
/// terminal Teardown event on connection drop.
#[test]
fn connection_drop_emits_teardown() {
    let (server, events) = start(60);
    let mut client = client_for(&server);
    let d = client.describe().unwrap();
    let control_url = resolve_control(&d.base, CONTROL);
    client.setup(&control_url, 5050).unwrap();
    let sid = client.session().unwrap().id.clone();
    client.play().unwrap();
    assert!(matches!(next_event(&events), ServerEvent::Play { .. }));

    drop(client); // TCP FIN, no TEARDOWN
    assert_eq!(next_event(&events), ServerEvent::Teardown { session: sid });
    server.shutdown();
}
