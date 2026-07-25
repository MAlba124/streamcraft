//! RTSP client framing tests over a real loopback `TcpStream` pair: a
//! scripted server thread reads the client's requests (asserting on their
//! shape) and answers canned RFC 2326 transcripts — DESCRIBE→SETUP→PLAY
//! happy path, 401→Basic/Digest retries (with hash values pinned per
//! RFC 2617 §3.2.2), keepalive, TEARDOWN.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::JoinHandle;

use sc_rtsp::client::{RtspClient, RtspError};
use sc_rtsp::sdp::MediaKind;

/// A connected loopback socket pair: (client end, server end).
fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let client = TcpStream::connect(addr).unwrap();
    let (server, _) = listener.accept().unwrap();
    (client, server)
}

/// One request as the scripted server sees it.
struct Req {
    method: String,
    uri: String,
    headers: Vec<(String, String)>,
}

impl Req {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    fn cseq(&self) -> &str {
        self.header("CSeq").expect("request without CSeq (§12.17)")
    }
}

/// Read one request head off the socket (our client never sends bodies in
/// these tests). `carry` holds bytes read past the previous request.
fn read_req(stream: &mut TcpStream, carry: &mut Vec<u8>) -> Req {
    loop {
        if let Some(pos) = carry.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8(carry[..pos].to_vec()).unwrap();
            carry.drain(..pos + 4);
            let mut lines = head.lines();
            let request_line = lines.next().unwrap();
            let mut parts = request_line.split(' ');
            let method = parts.next().unwrap().to_string();
            let uri = parts.next().unwrap().to_string();
            assert_eq!(parts.next(), Some("RTSP/1.0"), "§6.1 request line version");
            let headers = lines
                .map(|l| {
                    let (k, v) = l.split_once(':').expect("header line");
                    (k.trim().to_string(), v.trim().to_string())
                })
                .collect();
            return Req {
                method,
                uri,
                headers,
            };
        }
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk).unwrap();
        assert!(n > 0, "client closed while a request was expected");
        carry.extend_from_slice(&chunk[..n]);
    }
}

/// Send a response, echoing the request's CSeq (§12.17). `head` lines are
/// LF-separated for readability here and joined with CRLF on the wire;
/// `body` gets a Content-Length.
fn respond(stream: &mut TcpStream, req: &Req, head: &str, body: &str) {
    let mut out = format!(
        "RTSP/1.0 {}\r\nCSeq: {}\r\n",
        head.lines().next().unwrap(),
        req.cseq()
    );
    for line in head.lines().skip(1) {
        out.push_str(line);
        out.push_str("\r\n");
    }
    if !body.is_empty() {
        out.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    out.push_str("\r\n");
    out.push_str(body);
    stream.write_all(out.as_bytes()).unwrap();
}

/// Run a scripted server on its own thread; panics inside propagate through
/// `join` at the end of each test.
fn server(
    mut stream: TcpStream,
    script: impl FnOnce(&mut TcpStream) + Send + 'static,
) -> JoinHandle<()> {
    std::thread::spawn(move || script(&mut stream))
}

const SDP_BODY: &str = "v=0\r\n\
    o=- 1 1 IN IP4 127.0.0.1\r\n\
    s=loopback\r\n\
    t=0 0\r\n\
    a=control:*\r\n\
    m=video 0 RTP/AVP 96\r\n\
    a=rtpmap:96 H264/90000\r\n\
    a=fmtp:96 packetization-mode=1;sprop-parameter-sets=Z0LAHtkA8SJq,aMuDyyA=\r\n\
    a=control:trackID=0\r\n\
    m=audio 0 RTP/AVP 97\r\n\
    a=rtpmap:97 opus/48000/2\r\n\
    a=control:trackID=1\r\n";

/// The full unauthenticated session: DESCRIBE (with Content-Base driving
/// control resolution) → SETUP ×2 (Session established then echoed) → PLAY
/// (aggregate, on the base) → GET_PARAMETER keepalive → TEARDOWN.
#[test]
fn describe_setup_play_keepalive_teardown() {
    let (c, s) = pair();
    let handle = server(s, |s| {
        let mut carry = Vec::new();

        let req = read_req(s, &mut carry);
        assert_eq!(req.method, "DESCRIBE");
        assert_eq!(req.uri, "rtsp://127.0.0.1/test");
        // §10.2: the client asks for the description format it understands.
        assert_eq!(req.header("Accept"), Some("application/sdp"));
        assert_eq!(req.header("Session"), None, "no session before SETUP");
        respond(
            s,
            &req,
            "200 OK\nContent-Base: rtsp://127.0.0.1/test/\nContent-Type: application/sdp",
            SDP_BODY,
        );

        let req = read_req(s, &mut carry);
        assert_eq!(req.method, "SETUP");
        // §C.1.1: relative control resolved against Content-Base.
        assert_eq!(req.uri, "rtsp://127.0.0.1/test/trackID=0");
        // §12.39: our unicast UDP offer with the chosen port pair.
        assert_eq!(
            req.header("Transport"),
            Some("RTP/AVP;unicast;client_port=5000-5001")
        );
        respond(
            s,
            &req,
            "200 OK\nSession: 0xDEAD;timeout=30\n\
             Transport: RTP/AVP;unicast;client_port=5000-5001;server_port=6000-6001;source=127.0.0.9",
            "",
        );

        let req = read_req(s, &mut carry);
        assert_eq!(req.method, "SETUP");
        assert_eq!(req.uri, "rtsp://127.0.0.1/test/trackID=1");
        // §12.37: the session id must come back on related requests.
        assert_eq!(req.header("Session"), Some("0xDEAD"));
        respond(
            s,
            &req,
            "200 OK\nSession: 0xDEAD;timeout=30\n\
             Transport: RTP/AVP;unicast;client_port=5002-5003;server_port=6002-6003",
            "",
        );

        let req = read_req(s, &mut carry);
        assert_eq!(req.method, "PLAY");
        // §10.5 aggregate: PLAY goes to the presentation base URL.
        assert_eq!(req.uri, "rtsp://127.0.0.1/test/");
        assert_eq!(req.header("Session"), Some("0xDEAD"));
        respond(s, &req, "200 OK\nSession: 0xDEAD", "");

        let req = read_req(s, &mut carry);
        // §10.8: GET_PARAMETER with no body as a liveness ping.
        assert_eq!(req.method, "GET_PARAMETER");
        assert_eq!(req.header("Session"), Some("0xDEAD"));
        assert_eq!(req.header("Content-Length"), None, "keepalive has no body");
        respond(s, &req, "200 OK\nSession: 0xDEAD", "");

        let req = read_req(s, &mut carry);
        assert_eq!(req.method, "TEARDOWN");
        assert_eq!(req.header("Session"), Some("0xDEAD"));
        respond(s, &req, "200 OK", "");
    });

    let mut client = RtspClient::from_stream(c, "rtsp://127.0.0.1/test");

    let d = client.describe().unwrap();
    assert_eq!(d.base, "rtsp://127.0.0.1/test/");
    assert_eq!(d.sdp.media.len(), 2);
    assert_eq!(d.sdp.media[0].kind, MediaKind::Video);
    assert_eq!(d.raw, SDP_BODY);

    let video_ctrl =
        sc_rtsp::client::resolve_control(&d.base, d.sdp.media[0].control.as_deref().unwrap());
    let t = client.setup(&video_ctrl, 5000).unwrap();
    assert_eq!(t.server_port, Some((6000, 6001)));
    assert_eq!(t.client_port, Some((5000, 5001)));
    assert_eq!(t.source.as_deref(), Some("127.0.0.9"));
    let session = client.session().unwrap().clone();
    assert_eq!(session.id, "0xDEAD");
    assert_eq!(session.timeout_secs, 30);

    let audio_ctrl =
        sc_rtsp::client::resolve_control(&d.base, d.sdp.media[1].control.as_deref().unwrap());
    let t = client.setup(&audio_ctrl, 5002).unwrap();
    assert_eq!(t.server_port, Some((6002, 6003)));

    client.play().unwrap();
    client.keepalive().unwrap();
    client.teardown().unwrap();
    assert!(client.session().is_none(), "TEARDOWN forgets the session");

    handle.join().unwrap();
}

/// 401 → Digest retry, legacy no-qop (RFC 2069 compatibility) form. The
/// expected `response` is pinned from hand-computed hashes (RFC 2617
/// §3.2.2, verified with coreutils md5sum):
///
/// - `HA1 = MD5("mufasa:streamcraft:circle-of-life")` (§3.2.2.2:
///   `A1 = unq(username) ":" unq(realm) ":" passwd`)
///   = `c8059ca7382ba5fb48f2144d7efccda7`
/// - `HA2 = MD5("DESCRIBE:rtsp://127.0.0.1/test")` (§3.2.2.3:
///   `A2 = Method ":" digest-uri`)
///   = `6c1c54e312b25c4073fe4f9d084c290c`
/// - `response = MD5(HA1 ":" "1bcf5417a2" ":" HA2)` (§3.2.2.1 without qop)
///   = `f2c76514e2e423f6d822550dae27029a`
#[test]
fn digest_401_retry_no_qop() {
    let (c, s) = pair();
    let handle = server(s, |s| {
        let mut carry = Vec::new();

        let req = read_req(s, &mut carry);
        assert_eq!(req.method, "DESCRIBE");
        assert_eq!(req.header("Authorization"), None, "first try is bare");
        respond(
            s,
            &req,
            "401 Unauthorized\nWWW-Authenticate: Digest realm=\"streamcraft\", nonce=\"1bcf5417a2\"",
            "",
        );

        let req = read_req(s, &mut carry);
        assert_eq!(req.method, "DESCRIBE", "same request, retried once");
        let auth = req
            .header("Authorization")
            .expect("retry must authenticate");
        let auth = auth.strip_prefix("Digest ").expect("Digest scheme");
        let get = |name: &str| {
            auth.split(',').find_map(|p| {
                let (k, v) = p.trim().split_once('=')?;
                (k == name).then(|| v.trim().trim_matches('"').to_string())
            })
        };
        assert_eq!(get("username").as_deref(), Some("mufasa"));
        assert_eq!(get("realm").as_deref(), Some("streamcraft"));
        assert_eq!(get("nonce").as_deref(), Some("1bcf5417a2"));
        // §3.2.2: digest-uri is the Request-URI of the request line.
        assert_eq!(get("uri").as_deref(), Some("rtsp://127.0.0.1/test"));
        // No qop offered → no qop/nc/cnonce in the answer (§3.2.2.1).
        assert_eq!(get("qop"), None);
        assert_eq!(get("nc"), None);
        assert_eq!(
            get("response").as_deref(),
            Some("f2c76514e2e423f6d822550dae27029a"),
            "no-qop request-digest per §3.2.2.1"
        );
        respond(
            s,
            &req,
            "200 OK\nContent-Type: application/sdp",
            "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=x\r\nt=0 0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n",
        );
    });

    let mut client = RtspClient::from_stream(c, "rtsp://127.0.0.1/test");
    client.set_credentials("mufasa", "circle-of-life");
    let d = client.describe().unwrap();
    assert_eq!(d.sdp.media.len(), 1);
    handle.join().unwrap();
}

/// 401 → Digest retry with `qop="auth"` (§3.2.2.1's full form), cnonce
/// pinned. Expected `response`, hand-computed (md5sum) from the same
/// HA1/HA2 as [`digest_401_retry_no_qop`]:
///
/// - `response = MD5(HA1 ":1bcf5417a2:00000001:0a4f113b:auth:" HA2)`
///   = `87fcd2d4c74420aa052f1c2be86e7582`
///
/// A follow-up request must reuse the nonce with `nc=00000002` (§3.2.2:
/// the nonce-count increments per request; RFC 2617 §3.3 session reuse).
#[test]
fn digest_401_retry_qop_auth_and_nc_increments() {
    let (c, s) = pair();
    let handle = server(s, |s| {
        let mut carry = Vec::new();

        let req = read_req(s, &mut carry);
        respond(
            s,
            &req,
            "401 Unauthorized\nWWW-Authenticate: Digest realm=\"streamcraft\", \
             nonce=\"1bcf5417a2\", qop=\"auth\", opaque=\"cafef00d\"",
            "",
        );

        let req = read_req(s, &mut carry);
        let auth = req.header("Authorization").unwrap().to_string();
        assert!(auth.starts_with("Digest "));
        // §3.2.2 grammar: qop and nc are unquoted tokens; cnonce/response/
        // opaque are quoted-strings; opaque returns unchanged (§3.2.1).
        assert!(auth.contains("qop=auth"), "got {auth}");
        assert!(auth.contains("nc=00000001"), "got {auth}");
        assert!(auth.contains("cnonce=\"0a4f113b\""), "got {auth}");
        assert!(auth.contains("opaque=\"cafef00d\""), "got {auth}");
        assert!(
            auth.contains("response=\"87fcd2d4c74420aa052f1c2be86e7582\""),
            "qop=auth request-digest per §3.2.2.1, got {auth}"
        );
        respond(
            s,
            &req,
            "200 OK\nContent-Type: application/sdp",
            "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=x\r\nt=0 0\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H264/90000\r\n",
        );

        // The next request authenticates preemptively with the same nonce
        // and a bumped nonce-count.
        let req = read_req(s, &mut carry);
        assert_eq!(req.method, "OPTIONS");
        let auth = req.header("Authorization").unwrap();
        assert!(auth.contains("nonce=\"1bcf5417a2\""), "got {auth}");
        assert!(auth.contains("nc=00000002"), "got {auth}");
        respond(
            s,
            &req,
            "200 OK\nPublic: DESCRIBE, SETUP, PLAY, TEARDOWN",
            "",
        );
    });

    let mut client = RtspClient::from_stream(c, "rtsp://127.0.0.1/test");
    client.set_credentials("mufasa", "circle-of-life");
    client.set_cnonce("0a4f113b");
    client.describe().unwrap();
    let methods = client.options().unwrap();
    assert_eq!(methods, vec!["DESCRIBE", "SETUP", "PLAY", "TEARDOWN"]);
    handle.join().unwrap();
}

/// 401 → Basic retry (RFC 2617 §2): `base64("mufasa:circle-of-life")` =
/// `bXVmYXNhOmNpcmNsZS1vZi1saWZl` (pinned; verified with coreutils base64).
#[test]
fn basic_401_retry() {
    let (c, s) = pair();
    let handle = server(s, |s| {
        let mut carry = Vec::new();

        let req = read_req(s, &mut carry);
        assert_eq!(req.header("Authorization"), None);
        respond(
            s,
            &req,
            "401 Unauthorized\nWWW-Authenticate: Basic realm=\"streamcraft\"",
            "",
        );

        let req = read_req(s, &mut carry);
        assert_eq!(
            req.header("Authorization"),
            Some("Basic bXVmYXNhOmNpcmNsZS1vZi1saWZl"),
            "§2: base64 of userid:password"
        );
        respond(s, &req, "200 OK\nPublic: DESCRIBE", "");
    });

    let mut client = RtspClient::from_stream(c, "rtsp://127.0.0.1/test");
    client.set_credentials("mufasa", "circle-of-life");
    assert_eq!(client.options().unwrap(), vec!["DESCRIBE"]);
    handle.join().unwrap();
}

/// Retry-once semantics: a second 401 (wrong password) errors out instead
/// of looping; the server must see exactly two requests.
#[test]
fn second_401_is_an_error_not_a_loop() {
    let (c, s) = pair();
    let handle = server(s, |s| {
        let mut carry = Vec::new();
        for _ in 0..2 {
            let req = read_req(s, &mut carry);
            respond(
                s,
                &req,
                "401 Unauthorized\nWWW-Authenticate: Digest realm=\"r\", nonce=\"n\"",
                "",
            );
        }
        // A third request would block forever and time the test out; the
        // socket closing after two is the assertion.
    });

    let mut client = RtspClient::from_stream(c, "rtsp://127.0.0.1/test");
    client.set_credentials("mufasa", "wrong-password");
    match client.describe() {
        Err(RtspError::Unauthorized(_)) => {}
        other => panic!("expected Unauthorized, got {other:?}"),
    }
    handle.join().unwrap();
}

/// Without credentials a 401 surfaces as Unauthorized with no retry.
#[test]
fn no_credentials_means_no_retry() {
    let (c, s) = pair();
    let handle = server(s, |s| {
        let mut carry = Vec::new();
        let req = read_req(s, &mut carry);
        respond(
            s,
            &req,
            "401 Unauthorized\nWWW-Authenticate: Digest realm=\"r\", nonce=\"n\"",
            "",
        );
    });

    let mut client = RtspClient::from_stream(c, "rtsp://127.0.0.1/test");
    match client.describe() {
        Err(RtspError::Unauthorized(_)) => {}
        other => panic!("expected Unauthorized, got {other:?}"),
    }
    handle.join().unwrap();
}

/// A response whose CSeq does not echo the request's is a protocol error
/// (§12.17 pairs every request with the same-numbered response).
#[test]
fn cseq_mismatch_is_a_protocol_error() {
    let (c, s) = pair();
    let handle = server(s, |s| {
        let mut carry = Vec::new();
        let _req = read_req(s, &mut carry);
        s.write_all(b"RTSP/1.0 200 OK\r\nCSeq: 42\r\n\r\n").unwrap();
    });

    let mut client = RtspClient::from_stream(c, "rtsp://127.0.0.1/test");
    match client.options() {
        Err(RtspError::Protocol(m)) => assert!(m.contains("CSeq"), "got {m}"),
        other => panic!("expected Protocol error, got {other:?}"),
    }
    handle.join().unwrap();
}

/// Non-2xx statuses surface as errors from the method helpers.
#[test]
fn non_2xx_status_is_an_error() {
    let (c, s) = pair();
    let handle = server(s, |s| {
        let mut carry = Vec::new();
        let req = read_req(s, &mut carry);
        respond(s, &req, "454 Session Not Found", "");
    });

    let mut client = RtspClient::from_stream(c, "rtsp://127.0.0.1/test");
    match client.play() {
        Err(RtspError::Status(454, reason)) => assert_eq!(reason, "Session Not Found"),
        other => panic!("expected Status(454), got {other:?}"),
    }
    handle.join().unwrap();
}
