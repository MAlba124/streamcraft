//! A blocking RTSP 1.0 server (RFC 2326) over `std::net::TcpListener`.
//!
//! The split of responsibilities is the app-as-controller convention: this
//! module owns the **protocol** — message framing (§4), the method set
//! (§10), the per-session state machine (Appendix A.2 Init→Ready→Playing),
//! Transport negotiation (§12.39), session identifiers and timeouts
//! (§3.4/§12.37), and the error catalogue (§7.1.1/§11) — while the app owns
//! the **pipeline**: it supplies the presentation description served by
//! DESCRIBE ([`ServerConfig::sdp`]) and the real RTP/RTCP sockets, and it
//! learns *when and where* to stream through [`ServerEvent`]s on an
//! `mpsc::Sender` — [`ServerEvent::Play`] carries the destination the
//! client offered in SETUP's `client_port` (§12.39) combined with the
//! control connection's peer IP (§12.39 `destination`: "A server SHOULD
//! not allow a client to direct media streams to an address that differs
//! from the address commands are coming from" — we never honor
//! `destination`, closing that remote-controlled-DoS door by construction).
//!
//! **v1 boundaries**, chosen small and documented rather than half-built:
//!
//! - one media stream per presentation: [`ServerConfig::control`] names the
//!   single `a=control` track; a SETUP for any other URI gets 459 Aggregate
//!   Operation Not Allowed (§11.3.10 — with exactly one stream, the only
//!   non-aggregate URL is `<presentation>/<control>`);
//! - UDP unicast only: a Transport offer with no `RTP/AVP` (lower-transport
//!   UDP, the §12.39 default) unicast spec — e.g. `RTP/AVP/TCP` interleaving
//!   or `multicast` — is refused with 461 Unsupported Transport (§11.3.12);
//! - no aggregate multi-track SETUP, no RECORD/ANNOUNCE (501, §10 Table 2
//!   note: "If a server does not support a particular method, it MUST
//!   return '501 Not Implemented'"), no authentication;
//! - a session is bound to the control connection that created it: RFC 2326
//!   §1.2 lets one RTSP session span several transport connections, but v1
//!   answers a session id from any other connection with 454;
//! - PLAY's Range header is ignored (media timing belongs to the app); the
//!   response carries no Range/RTP-Info.
//!
//! Liveness (§12.37/A.2): every request naming the session refreshes its
//! timer; GET_PARAMETER with no body is the sanctioned ping (§10.8). When
//! no such request arrives within [`ServerConfig::timeout_secs`] — the
//! value advertised as `Session: id;timeout=N` (§12.37: the parameter "is
//! only allowed in a response header. The server uses it to indicate ...
//! how long the server is prepared to wait between RTSP commands before
//! closing the session") — the server reverts to Init as A.2 permits and
//! emits [`ServerEvent::Teardown`], as it also does when the connection
//! drops with a live session, so the app can always pair every Play with a
//! terminal Teardown.
//!
//! Threading is deliberately boring: one accept thread, one thread per
//! connection, session state owned by its connection thread, a shared
//! atomic shutdown flag. Connection reads poll with a short timeout so
//! session expiry and [`RtspServer::shutdown`] are noticed without any
//! signaling machinery.

// Protocol-layer blocking sockets, sanctioned: the RTSP control connection
// runs on app/controller threads (spec: no-bins), never inside an element's
// scheduler group — the workspace element-IO lint does not apply here.
#![allow(clippy::disallowed_methods)]

use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::md5::md5_hex;

/// Cap on a request head, so a peer that never terminates its headers
/// can't make us buffer unboundedly (same guard as the client's).
const MAX_HEAD_BYTES: usize = 64 * 1024;

/// Cap on a request body (SET_PARAMETER/GET_PARAMETER payloads are tiny;
/// past this is a broken or hostile peer). Over the cap we answer 413
/// Request Entity Too Large (§7.1.1) and drop the connection, since the
/// unread body would desynchronize the framing.
const MAX_BODY_BYTES: usize = 64 * 1024;

/// Connection read poll interval: reads block at most this long so the
/// per-connection thread can check session expiry (§12.37) and server
/// shutdown between requests.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Write timeout on connection sockets, so a wedged peer fails its own
/// connection instead of hanging the thread.
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);

/// An RTSP session identifier (§3.4: an opaque string, chosen by the
/// server, at least eight octets).
pub type SessionId = String;

/// What the server serves and the transport facts it advertises. The app
/// owns the actual media sockets; this is the protocol-visible description
/// of them.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// The presentation description returned by DESCRIBE (§10.2), verbatim,
    /// as `application/sdp`.
    pub sdp: String,
    /// The single media's `a=control` name (RFC 2326 §C.1.1), e.g.
    /// `trackID=0`. SETUP must address `<presentation>/<control>`.
    pub control: String,
    /// The `server_port=lo-hi` pair announced in the SETUP Transport
    /// response (§12.39) — the app's RTP/RTCP source ports. Purely
    /// declarative here: this module never opens them.
    pub server_port: (u16, u16),
    /// Session inactivity timeout in seconds, advertised as
    /// `Session: id;timeout=N` (§12.37, default 60). A session that
    /// receives no request for this long is torn down (A.2) with a
    /// [`ServerEvent::Teardown`].
    pub timeout_secs: u64,
}

impl ServerConfig {
    /// A config with the §12.37 default timeout (60 s) and a placeholder
    /// server_port pair; override the fields the app actually knows.
    pub fn new(sdp: impl Into<String>, control: impl Into<String>) -> Self {
        ServerConfig {
            sdp: sdp.into(),
            control: control.into(),
            server_port: (6970, 6971),
            timeout_secs: 60,
        }
    }
}

/// Pipeline-relevant protocol transitions, delivered to the app's mpsc
/// channel. Send failures (receiver gone) are ignored — the protocol side
/// keeps answering correctly either way.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ServerEvent {
    /// PLAY succeeded (§10.5): start (or resume) sending RTP to
    /// `client_rtp` and RTCP to `client_rtcp` — the SETUP `client_port`
    /// pair (§12.39) at the control connection's peer address. Re-emitted
    /// if the client PLAYs again while already playing (A.2 allows
    /// Playing→PLAY→Playing).
    Play {
        session: SessionId,
        client_rtp: SocketAddr,
        client_rtcp: SocketAddr,
    },
    /// PAUSE succeeded (§10.6): halt delivery, keep resources.
    Pause { session: SessionId },
    /// The session ended: TEARDOWN (§10.7), session timeout (§12.37/A.2),
    /// or the control connection dropped while the session was live.
    /// Emitted exactly once per session.
    Teardown { session: SessionId },
}

/// State shared between the server handle, the accept thread and the
/// connection threads: the shutdown flag. Connection reads poll on
/// [`POLL_INTERVAL`], so no wake-up plumbing is needed for them — each
/// thread observes the flag within one tick, drops its socket (sending
/// FIN) and exits.
struct Shared {
    /// Set by [`RtspServer::shutdown`]; accept and connection loops exit
    /// when they observe it.
    shutdown: AtomicBool,
}

/// A listening RTSP server: an accept thread plus one thread per client
/// connection. Dropping the handle (or calling [`RtspServer::shutdown`])
/// stops accepting, closes every connection and joins all threads.
pub struct RtspServer {
    local: SocketAddr,
    shared: Arc<Shared>,
    accept: Option<JoinHandle<()>>,
}

impl RtspServer {
    /// Bind and start serving. `events` receives the [`ServerEvent`]
    /// transitions the app acts on.
    pub fn bind(
        addr: SocketAddr,
        config: ServerConfig,
        events: Sender<ServerEvent>,
    ) -> std::io::Result<RtspServer> {
        let listener = TcpListener::bind(addr)?;
        let local = listener.local_addr()?;
        let shared = Arc::new(Shared {
            shutdown: AtomicBool::new(false),
        });
        let config = Arc::new(config);
        let accept = {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || accept_loop(listener, config, events, shared))
        };
        Ok(RtspServer {
            local,
            shared,
            accept: Some(accept),
        })
    }

    /// The bound address (with the OS-chosen port when bound to port 0).
    pub fn local_addr(&self) -> SocketAddr {
        self.local
    }

    /// Stop accepting, close client connections, join all threads. Live
    /// sessions emit a final [`ServerEvent::Teardown`] as their
    /// connections close. (Dropping the handle does the same.)
    pub fn shutdown(self) {
        // Drop runs the actual teardown; the method exists so callers can
        // say what they mean.
    }
}

impl Drop for RtspServer {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        // `accept` has no timeout; a loopback connect wakes it so it can
        // observe the flag. A wildcard bind isn't dialable as-is — aim the
        // wake-up at the loopback of the same address family.
        let ip = match self.local.ip() {
            IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
            IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
            ip => ip,
        };
        let _ = TcpStream::connect_timeout(
            &SocketAddr::new(ip, self.local.port()),
            Duration::from_secs(1),
        );
        // Connection threads notice the flag within one POLL_INTERVAL and
        // exit; the accept thread joins them before it returns.
        if let Some(h) = self.accept.take() {
            let _ = h.join();
        }
    }
}

/// Accept until shutdown; join the connection threads on the way out so
/// [`RtspServer::shutdown`] returning means everything stopped.
fn accept_loop(
    listener: TcpListener,
    config: Arc<ServerConfig>,
    events: Sender<ServerEvent>,
    shared: Arc<Shared>,
) {
    let mut workers: Vec<JoinHandle<()>> = Vec::new();
    for incoming in listener.incoming() {
        if shared.shutdown.load(Ordering::Acquire) {
            break;
        }
        let Ok(stream) = incoming else { continue };
        let config = Arc::clone(&config);
        let events = events.clone();
        let shared = Arc::clone(&shared);
        workers.push(std::thread::spawn(move || {
            serve_connection(stream, config, events, shared)
        }));
    }
    for w in workers {
        let _ = w.join();
    }
}

/// Per-session server state, Appendix A.2. `Init` is represented by the
/// absence of a session ([`Conn::session`] = `None`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    /// "Last SETUP received was successful ... or after playing, last
    /// PAUSE received was successful" (A.2).
    Ready,
    /// "Last PLAY received was successful ... Data is being sent" (A.2).
    Playing,
}

/// The one session a connection may own (v1 binds sessions to their
/// control connection).
struct SessionState {
    id: SessionId,
    state: State,
    client_rtp: SocketAddr,
    client_rtcp: SocketAddr,
    /// Refreshed by every request naming the session; drives the §12.37
    /// inactivity teardown.
    last_seen: Instant,
}

/// One parsed request (§6).
struct Request {
    method: String,
    uri: String,
    version: String,
    /// Header fields in arrival order (names case-insensitive, [H4.2]).
    headers: Vec<(String, String)>,
}

impl Request {
    /// First value of header `name`, case-insensitively ([H4.2]).
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// How the request's Session header relates to the connection's session.
enum SessionCheck {
    /// The request names the live session.
    Match,
    /// Neither a Session header nor a live session — state Init (A.2).
    Absent,
    /// A Session header that names nothing live (bogus, timed out, or
    /// omitted while a session exists — §12.37 requires the client to
    /// return the id on every related request).
    Mismatch,
}

/// What [`Conn::next_request`] produced.
enum Next {
    Request(Request),
    /// Poll timeout — no bytes owed; run the periodic checks.
    Tick,
    /// EOF, socket error, or an unrecoverable framing error (already
    /// answered where possible).
    Closed,
}

struct Conn {
    stream: TcpStream,
    peer: SocketAddr,
    /// Bytes read past the previous request (a peer may batch writes).
    inbuf: Vec<u8>,
    session: Option<SessionState>,
    config: Arc<ServerConfig>,
    events: Sender<ServerEvent>,
}

/// Serve one control connection until it closes or the server shuts down.
fn serve_connection(
    stream: TcpStream,
    config: Arc<ServerConfig>,
    events: Sender<ServerEvent>,
    shared: Arc<Shared>,
) {
    let Ok(peer) = stream.peer_addr() else { return };
    if stream.set_read_timeout(Some(POLL_INTERVAL)).is_err()
        || stream.set_write_timeout(Some(WRITE_TIMEOUT)).is_err()
    {
        return;
    }
    let mut conn = Conn {
        stream,
        peer,
        inbuf: Vec::new(),
        session: None,
        config,
        events,
    };
    loop {
        match conn.next_request() {
            Next::Request(req) => {
                if conn.handle(&req).is_err() {
                    break; // write failure: the connection is gone
                }
            }
            Next::Tick => {}
            Next::Closed => break,
        }
        if shared.shutdown.load(Ordering::Acquire) {
            break;
        }
        conn.expire_session();
    }
    // Connection drop with a live session: the app must still learn the
    // session ended (A.2 reverts to Init when the client goes away).
    conn.end_session();
}

impl Conn {
    /// Emit the terminal Teardown for the connection's session, if one is
    /// still live. Exactly-once: the session is consumed.
    fn end_session(&mut self) {
        if let Some(s) = self.session.take() {
            let _ = self.events.send(ServerEvent::Teardown { session: s.id });
        }
    }

    /// §12.37/A.2 inactivity teardown: no request has named the session
    /// for `timeout_secs` — revert to Init and tell the app. The control
    /// connection itself stays open; later use of the stale id gets 454
    /// ("has timed out", §11.3.5).
    fn expire_session(&mut self) {
        let expired = self
            .session
            .as_ref()
            .is_some_and(|s| s.last_seen.elapsed() >= Duration::from_secs(self.config.timeout_secs));
        if expired {
            self.end_session();
        }
    }

    /// Read one request: head to the blank line (§4: CRLF line ends, but
    /// bare LF tolerated), then a `Content-Length` body if declared
    /// (§4.4: "If this header field is not present, a value of zero is
    /// assumed"). Malformed framing is answered with 400 (or 413 over the
    /// body cap) and closes the connection, since the byte stream can no
    /// longer be trusted.
    fn next_request(&mut self) -> Next {
        loop {
            // Robustness: drop stray blank lines between requests.
            let lead = self
                .inbuf
                .iter()
                .take_while(|&&b| b == b'\r' || b == b'\n')
                .count();
            if lead > 0 {
                self.inbuf.drain(..lead);
            }

            if let Some(head_len) = head_end(&self.inbuf) {
                let head = String::from_utf8_lossy(&self.inbuf[..head_len]).into_owned();
                let req = match parse_head(&head) {
                    Ok(req) => req,
                    Err(why) => {
                        // §12.17 wants CSeq echoed on every response, but a
                        // request too broken to parse supplies none.
                        let _ = self.respond(400, "Bad Request", None, &[], why.as_bytes());
                        return Next::Closed;
                    }
                };
                let body_len = match req.header("Content-Length") {
                    Some(v) => match v.trim().parse::<usize>() {
                        Ok(n) => n,
                        Err(_) => {
                            let _ = self.respond(
                                400,
                                "Bad Request",
                                req.header("CSeq"),
                                &[],
                                b"unparsable Content-Length",
                            );
                            return Next::Closed;
                        }
                    },
                    None => 0,
                };
                if body_len > MAX_BODY_BYTES {
                    let _ = self.respond(
                        413,
                        "Request Entity Too Large",
                        req.header("CSeq"),
                        &[],
                        &[],
                    );
                    return Next::Closed;
                }
                if self.inbuf.len() >= head_len + body_len {
                    // The body (GET/SET_PARAMETER payloads) is consumed for
                    // framing but not interpreted in v1.
                    self.inbuf.drain(..head_len + body_len);
                    return Next::Request(req);
                }
                // Head parsed but body incomplete: fall through to read
                // more (the head is re-parsed next pass; state lives in
                // `inbuf`, so a poll tick in between loses nothing).
            } else if self.inbuf.len() > MAX_HEAD_BYTES {
                let _ = self.respond(400, "Bad Request", None, &[], b"request head too large");
                return Next::Closed;
            }

            let mut chunk = [0u8; 4096];
            match self.stream.read(&mut chunk) {
                Ok(0) => return Next::Closed,
                Ok(n) => self.inbuf.extend_from_slice(&chunk[..n]),
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    return Next::Tick;
                }
                Err(_) => return Next::Closed,
            }
        }
    }

    /// Dispatch one request. `Err` means the response could not be
    /// written and the connection should close.
    fn handle(&mut self, req: &Request) -> std::io::Result<()> {
        // §12.17: "This field MUST be present in all requests"; without it
        // the response could not be paired, so the request is malformed.
        let Some(cseq) = req.header("CSeq").map(str::to_owned) else {
            return self.respond(400, "Bad Request", None, &[], b"missing CSeq (RFC 2326 s12.17)");
        };
        let cseq = cseq.as_str();

        // §6.1 RTSP-Version. A well-formed but foreign version is 505 (the
        // §7.1.1 catalogue), not a framing failure — keep the connection.
        if req.version != "RTSP/1.0" {
            return self.respond(505, "RTSP Version Not Supported", Some(cseq), &[], &[]);
        }

        // Liveness (§12.37/A.2): any request naming the live session
        // counts as wellness and refreshes its timer.
        if let (Some(h), Some(s)) = (req.header("Session"), self.session.as_mut()) {
            if session_header_id(h) == s.id {
                s.last_seen = Instant::now();
            }
        }

        // §6.1: methods are tokens, matched case-sensitively ([H5.1.1]).
        match req.method.as_str() {
            "OPTIONS" => self.options(cseq),
            "DESCRIBE" => self.describe(req, cseq),
            "SETUP" => self.setup(req, cseq),
            "PLAY" => self.play(req, cseq),
            "PAUSE" => self.pause(req, cseq),
            "TEARDOWN" => self.teardown(req, cseq),
            "GET_PARAMETER" => self.get_parameter(req, cseq),
            // §10 (Table 2 note): "If a server does not support a
            // particular method, it MUST return '501 Not Implemented'" —
            // covers RECORD/ANNOUNCE/SET_PARAMETER/REDIRECT and any
            // extension-method (§6.1) alike.
            _ => self.respond(501, "Not Implemented", Some(cseq), &[], &[]),
        }
    }

    /// OPTIONS (§10.1): advertise the implemented set in `Public`; "does
    /// not influence server state", so no session is required (a matching
    /// one was already refreshed above).
    fn options(&mut self, cseq: &str) -> std::io::Result<()> {
        self.respond(
            200,
            "OK",
            Some(cseq),
            &[(
                "Public",
                "OPTIONS, DESCRIBE, SETUP, PLAY, PAUSE, TEARDOWN, GET_PARAMETER",
            )],
            &[],
        )
    }

    /// DESCRIBE (§10.2): serve the configured SDP. Content-Base (§12.11 /
    /// [H14.11]) names the presentation base the client resolves
    /// `a=control` against (§C.1.1 rule 1) — the request URL with a
    /// trailing slash, so relative controls append cleanly.
    fn describe(&mut self, req: &Request, cseq: &str) -> std::io::Result<()> {
        // §10.2: the client "may use the Accept header to specify the
        // description formats that the client understands"; we only speak
        // SDP — anything that excludes it is 406 ([H10.4.7] via §7.1.1).
        if let Some(accept) = req.header("Accept") {
            let sdp_ok = accept
                .split(',')
                .any(|t| t.trim().starts_with("application/sdp") || t.trim().starts_with("*/*"));
            if !sdp_ok {
                return self.respond(406, "Not Acceptable", Some(cseq), &[], &[]);
            }
        }
        let base = if req.uri.ends_with('/') {
            req.uri.clone()
        } else {
            format!("{}/", req.uri)
        };
        let body = self.config.sdp.clone();
        self.respond(
            200,
            "OK",
            Some(cseq),
            &[
                ("Content-Base", base.as_str()),
                ("Content-Type", "application/sdp"),
            ],
            body.as_bytes(),
        )
    }

    /// SETUP (§10.4): negotiate transport for the single configured track
    /// and mint the session (§3.4/§12.37).
    fn setup(&mut self, req: &Request, cseq: &str) -> std::io::Result<()> {
        match (req.header("Session"), &self.session) {
            // §10.4: a SETUP on a playing/ready session asks to change
            // transport parameters, "which a server MAY allow. If it does
            // not allow this, it MUST respond with error '455 Method Not
            // Valid In This State'." v1 does not allow it.
            (Some(h), Some(s)) if session_header_id(h) == s.id => {
                return self.method_not_valid(cseq);
            }
            // A session id we don't know: 454 (§11.3.5).
            (Some(_), _) => return self.session_not_found(cseq),
            // v1: one session per connection; a second bare SETUP while
            // one is live is a state error, not a new user session.
            (None, Some(_)) => return self.method_not_valid(cseq),
            (None, None) => {}
        }

        // v1 serves exactly one stream: the only SETUP-able URL is
        // `<presentation>/<control>` (§C.1.1); everything else — notably
        // the presentation URL itself — is an aggregate, and "the
        // requested method may not be applied on the URL in question
        // since it is an aggregate (presentation) URL" (§11.3.10).
        let last_segment = req.uri.trim_end_matches('/').rsplit('/').next();
        if last_segment != Some(self.config.control.as_str()) {
            return self.respond(459, "Aggregate Operation Not Allowed", Some(cseq), &[], &[]);
        }

        // §12.39 negotiation. A SETUP without a Transport offer proposes
        // nothing we could select from — malformed, 400. An offer with no
        // spec we can serve is 461 (§11.3.12).
        let Some(offer) = req.header("Transport") else {
            return self.respond(400, "Bad Request", Some(cseq), &[], b"SETUP without Transport");
        };
        let Some((rtp_port, rtcp_port)) = choose_transport(offer) else {
            return self.respond(461, "Unsupported Transport", Some(cseq), &[], &[]);
        };

        let id = generate_session_id(&self.peer);
        // §12.37: timeout "is only allowed in a response header" — this is
        // the one place it is announced.
        let session_hdr = format!("{id};timeout={}", self.config.timeout_secs);
        // §12.39: "the server MUST return a single option which was
        // actually chosen", echoing the client's ports and naming ours.
        let transport_hdr = format!(
            "RTP/AVP;unicast;client_port={rtp_port}-{rtcp_port};server_port={}-{}",
            self.config.server_port.0, self.config.server_port.1
        );
        self.session = Some(SessionState {
            id,
            state: State::Ready, // A.2: Init --SETUP--> Ready
            client_rtp: SocketAddr::new(self.peer.ip(), rtp_port),
            client_rtcp: SocketAddr::new(self.peer.ip(), rtcp_port),
            last_seen: Instant::now(),
        });
        self.respond(
            200,
            "OK",
            Some(cseq),
            &[
                ("Session", session_hdr.as_str()),
                ("Transport", transport_hdr.as_str()),
            ],
            &[],
        )
    }

    /// PLAY (§10.5): valid in Ready and Playing (A.2); tells the app to
    /// start delivery to the SETUP-negotiated destination.
    fn play(&mut self, req: &Request, cseq: &str) -> std::io::Result<()> {
        match self.session_check(req) {
            SessionCheck::Match => {}
            // No session at all: state Init, where A.2 lists no PLAY
            // transition — 455 (§11.3.6).
            SessionCheck::Absent => return self.method_not_valid(cseq),
            SessionCheck::Mismatch => return self.session_not_found(cseq),
        }
        let s = self.session.as_mut().expect("checked Match");
        s.state = State::Playing; // A.2: Ready|Playing --PLAY--> Playing
        let (id, session_hdr) = (s.id.clone(), s.id.clone());
        let (rtp, rtcp) = (s.client_rtp, s.client_rtcp);
        let _ = self.events.send(ServerEvent::Play {
            session: id,
            client_rtp: rtp,
            client_rtcp: rtcp,
        });
        self.respond(200, "OK", Some(cseq), &[("Session", session_hdr.as_str())], &[])
    }

    /// PAUSE (§10.6): "causes the stream delivery to be interrupted
    /// (halted) temporarily"; valid only while Playing (A.2's Ready row
    /// has no PAUSE transition).
    fn pause(&mut self, req: &Request, cseq: &str) -> std::io::Result<()> {
        match self.session_check(req) {
            SessionCheck::Match => {}
            SessionCheck::Absent => return self.method_not_valid(cseq),
            SessionCheck::Mismatch => return self.session_not_found(cseq),
        }
        let s = self.session.as_mut().expect("checked Match");
        if s.state != State::Playing {
            return self.method_not_valid(cseq);
        }
        s.state = State::Ready; // A.2: Playing --PAUSE--> Ready
        let (id, session_hdr) = (s.id.clone(), s.id.clone());
        let _ = self.events.send(ServerEvent::Pause { session: id });
        self.respond(200, "OK", Some(cseq), &[("Session", session_hdr.as_str())], &[])
    }

    /// TEARDOWN (§10.7): "stops the stream delivery ..., freeing the
    /// resources associated with it"; the session id "is no longer
    /// valid". Valid in any state that has a session (A.2).
    fn teardown(&mut self, req: &Request, cseq: &str) -> std::io::Result<()> {
        match self.session_check(req) {
            SessionCheck::Match => {}
            // With nothing set up there is nothing to tear down; §11.3.5
            // reads a missing Session header as 454 territory.
            SessionCheck::Absent | SessionCheck::Mismatch => {
                return self.session_not_found(cseq);
            }
        }
        self.end_session(); // A.2: * --TEARDOWN--> Init; emits the event
        self.respond(200, "OK", Some(cseq), &[], &[])
    }

    /// GET_PARAMETER (§10.8): "with no entity body may be used to test
    /// client or server liveness ('ping')". v1 implements exactly the
    /// ping: 200 with an empty body; the matching-session refresh already
    /// happened in [`Conn::handle`]. Works session-less too (§10.1-style
    /// probing needs no state), but a bogus session id is still 454.
    fn get_parameter(&mut self, req: &Request, cseq: &str) -> std::io::Result<()> {
        if let SessionCheck::Mismatch = self.session_check(req) {
            return self.session_not_found(cseq);
        }
        let session_hdr = self.session.as_ref().map(|s| s.id.clone());
        let headers: Vec<(&str, &str)> = match &session_hdr {
            Some(id) => vec![("Session", id.as_str())],
            None => vec![],
        };
        self.respond(200, "OK", Some(cseq), &headers, &[])
    }

    /// How the request's Session header relates to this connection's
    /// session (v1 scope: sessions never cross connections).
    fn session_check(&self, req: &Request) -> SessionCheck {
        match (req.header("Session"), &self.session) {
            (Some(h), Some(s)) if session_header_id(h) == s.id => SessionCheck::Match,
            (None, None) => SessionCheck::Absent,
            // Wrong id, a timed-out id, or an omitted id while a session
            // exists (§12.37: the client "MUST return it for any request
            // related to that session") — all §11.3.5.
            _ => SessionCheck::Mismatch,
        }
    }

    /// 454 Session Not Found (§11.3.5: "The RTSP session identifier in
    /// the Session header is missing, invalid, or has timed out").
    fn session_not_found(&mut self, cseq: &str) -> std::io::Result<()> {
        self.respond(454, "Session Not Found", Some(cseq), &[], &[])
    }

    /// 455 Method Not Valid In This State (§11.3.6), with the SHOULD-level
    /// Allow header listing what the current A.2 state accepts.
    fn method_not_valid(&mut self, cseq: &str) -> std::io::Result<()> {
        let allow = match self.session.as_ref().map(|s| s.state) {
            None => "OPTIONS, DESCRIBE, SETUP, GET_PARAMETER",
            Some(State::Ready) => "OPTIONS, DESCRIBE, PLAY, TEARDOWN, GET_PARAMETER",
            Some(State::Playing) => "OPTIONS, DESCRIBE, PLAY, PAUSE, TEARDOWN, GET_PARAMETER",
        };
        self.respond(
            455,
            "Method Not Valid In This State",
            Some(cseq),
            &[("Allow", allow)],
            &[],
        )
    }

    /// Write one response: §7.1 Status-Line, the §12.17 CSeq echo, extra
    /// headers, and a `Content-Length`-framed body when non-empty (§12.14:
    /// required "in all messages that carry content"; absent means zero).
    fn respond(
        &mut self,
        status: u16,
        reason: &str,
        cseq: Option<&str>,
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> std::io::Result<()> {
        let mut out = format!("RTSP/1.0 {status} {reason}\r\n");
        if let Some(c) = cseq {
            out.push_str(&format!("CSeq: {c}\r\n"));
        }
        for (k, v) in headers {
            out.push_str(&format!("{k}: {v}\r\n"));
        }
        if !body.is_empty() {
            out.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        out.push_str("\r\n");
        self.stream.write_all(out.as_bytes())?;
        if !body.is_empty() {
            self.stream.write_all(body)?;
        }
        Ok(())
    }
}

/// The session-id part of a request's Session header. §12.37's request
/// grammar is the bare id, but tolerate a client that reflects parameters
/// (e.g. the `;timeout=` it was handed) back at us.
fn session_header_id(value: &str) -> &str {
    value.split(';').next().unwrap_or("").trim()
}

/// Mint a session identifier (§3.4: "opaque strings ... MUST be chosen
/// randomly and MUST be at least eight octets long to make guessing it
/// more difficult"). Dependency-free source: MD5 over the wall clock, a
/// process-wide counter, the peer address and ASLR'd addresses, truncated
/// to 16 hex chars (8 octets of digest). Unguessable enough to avoid
/// accidental collisions and casual guessing; do not lean on session-id
/// secrecy as an authentication boundary (v1 has no auth at all).
fn generate_session_id(peer: &SocketAddr) -> SessionId {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seed = format!(
        "{}.{:09}:{}:{}:{:p}:{:p}",
        now.as_secs(),
        now.subsec_nanos(),
        n,
        peer,
        &n,        // stack address
        &COUNTER,  // static address
    );
    md5_hex(seed.as_bytes())[..16].to_string()
}

/// Pick the client ports from a SETUP Transport offer (§12.39):
/// "Transports are comma separated, listed in order of preference" — the
/// first spec this v1 can serve wins. Returns the `client_port` pair, or
/// `None` when no spec is acceptable (the caller answers 461, §11.3.12).
///
/// A spec is acceptable when all of:
/// - the transport-spec is `RTP/AVP` or `RTP/AVP/UDP` (§12.39: "For
///   RTP/AVP, the default [lower-transport] is UDP") — `RTP/AVP/TCP` and
///   anything else is out (v1 is UDP-only);
/// - it is not `multicast` and carries no `interleaved` channels. §12.39
///   defaults delivery to multicast when neither word appears, but it also
///   defines `client_port` as "the unicast RTP/RTCP port pair on which the
///   client has chosen to receive media data" — so a spec that names
///   `client_port` without saying `unicast` is read as the unicast request
///   it plainly is (ambiguity resolved toward interoperability);
/// - `client_port=lo-hi` is present (without it there is nowhere to send);
/// - `mode`, if given, includes PLAY (§12.39: "If not provided, the
///   default is PLAY"; v1 serves nothing else).
fn choose_transport(offer: &str) -> Option<(u16, u16)> {
    // §12.39's mode value is a quoted list (`mode="PLAY,RECORD"`), so both
    // the spec separator (,) and the parameter separator (;) must respect
    // quoting.
    split_unquoted(offer, ',')
        .into_iter()
        .find_map(|spec| accept_spec(spec.trim()))
}

/// One transport-spec (§12.39 grammar:
/// `transport-protocol/profile[/lower-transport] *parameter`).
fn accept_spec(spec: &str) -> Option<(u16, u16)> {
    let mut params = split_unquoted(spec, ';').into_iter();
    let proto = params.next()?.trim();
    if !(proto.eq_ignore_ascii_case("RTP/AVP") || proto.eq_ignore_ascii_case("RTP/AVP/UDP")) {
        return None;
    }
    let mut client_port = None;
    for p in params {
        let p = p.trim();
        if p.eq_ignore_ascii_case("multicast") {
            return None; // v1 is unicast-only
        }
        if let Some((k, v)) = p.split_once('=') {
            let (k, v) = (k.trim(), v.trim());
            if k.eq_ignore_ascii_case("client_port") {
                client_port = parse_port_pair(v);
            } else if k.eq_ignore_ascii_case("interleaved") {
                return None; // §10.12 TCP interleaving — not this transport
            } else if k.eq_ignore_ascii_case("mode") {
                // mode = <"> 1#mode <"> | Method (§12.39); default PLAY.
                let plays = v
                    .trim_matches('"')
                    .split(',')
                    .any(|m| m.trim().eq_ignore_ascii_case("PLAY"));
                if !plays {
                    return None;
                }
            }
        }
    }
    client_port
}

/// `lo-hi` port ranges (§12.39: "It is specified as a range, e.g.,
/// client_port=3456-3457"); a lone port is taken as `p-(p+1)` (RTP on the
/// even port, RTCP one above, per RTP convention).
fn parse_port_pair(v: &str) -> Option<(u16, u16)> {
    match v.split_once('-') {
        Some((lo, hi)) => Some((lo.trim().parse().ok()?, hi.trim().parse().ok()?)),
        None => {
            let p: u16 = v.trim().parse().ok()?;
            Some((p, p.checked_add(1)?))
        }
    }
}

/// Split on `sep` outside double quotes — §12.39 parameter values may be
/// quoted lists (`mode="PLAY,RECORD"`) whose commas are not separators.
fn split_unquoted(s: &str, sep: char) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut in_quotes = false;
    for (i, c) in s.char_indices() {
        if c == '"' {
            in_quotes = !in_quotes;
        } else if c == sep && !in_quotes {
            out.push(&s[start..i]);
            start = i + sep.len_utf8();
        }
    }
    out.push(&s[start..]);
    out
}

/// Index just past the request head's terminating blank line. Lines end
/// CRLF, but "receivers should be prepared to also interpret CR and LF by
/// themselves as line terminators" (§4) — `\n\n` is accepted too.
fn head_end(buf: &[u8]) -> Option<usize> {
    for (i, &b) in buf.iter().enumerate() {
        if b != b'\n' {
            continue;
        }
        if buf.get(i + 1) == Some(&b'\n') {
            return Some(i + 2); // "\n\n"
        }
        if buf.get(i + 1) == Some(&b'\r') && buf.get(i + 2) == Some(&b'\n') {
            return Some(i + 3); // "\n\r\n" — i.e. "\r\n\r\n" ends here
        }
    }
    None
}

/// Parse a request head: the §6.1 Request-Line
/// (`Method SP Request-URI SP RTSP-Version`) then header fields to the
/// blank line ([H4.2], including LWS continuation folding).
fn parse_head(head: &str) -> Result<Request, String> {
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split(' ');
    let (method, uri, version) = match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(m), Some(u), Some(v), None) if !m.is_empty() && !u.is_empty() => (m, u, v),
        _ => return Err(format!("malformed request line {request_line:?}")),
    };
    if !version.starts_with("RTSP/") {
        return Err(format!("not an RTSP request line: {request_line:?}"));
    }

    let mut headers: Vec<(String, String)> = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        // [H4.2] LWS folding: a line starting with SP/HT continues the
        // previous field value.
        if line.starts_with(' ') || line.starts_with('\t') {
            if let Some((_, v)) = headers.last_mut() {
                v.push(' ');
                v.push_str(line.trim());
            }
            continue;
        }
        let Some((k, v)) = line.split_once(':') else {
            return Err(format!("malformed header line {line:?}"));
        };
        headers.push((k.trim().to_string(), v.trim().to_string()));
    }
    Ok(Request {
        method: method.to_string(),
        uri: uri.to_string(),
        version: version.to_string(),
        headers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_udp_unicast_offer_is_accepted() {
        // The §12.39 example offer our own client sends.
        assert_eq!(
            choose_transport("RTP/AVP;unicast;client_port=4588-4589"),
            Some((4588, 4589))
        );
        // Explicit lower-transport spelling (§12.39: default UDP for
        // RTP/AVP, so both forms name the same transport).
        assert_eq!(
            choose_transport("RTP/AVP/UDP;unicast;client_port=4588-4589"),
            Some((4588, 4589))
        );
    }

    #[test]
    fn transport_tcp_and_multicast_are_rejected() {
        // v1 is UDP-unicast only: interleaved TCP (§10.12) → no spec.
        assert_eq!(choose_transport("RTP/AVP/TCP;unicast;interleaved=0-1"), None);
        // An interleaved parameter marks TCP intent even under RTP/AVP.
        assert_eq!(choose_transport("RTP/AVP;unicast;interleaved=0-1"), None);
        assert_eq!(choose_transport("RTP/AVP;multicast;ttl=127"), None);
        // Unknown protocol stacks are not guessed at.
        assert_eq!(choose_transport("RAW/RAW/UDP;unicast;client_port=1-2"), None);
    }

    #[test]
    fn transport_preference_order_picks_first_acceptable_spec() {
        // §12.39: "Transports are comma separated, listed in order of
        // preference" — the TCP offer is skipped, the UDP one chosen.
        assert_eq!(
            choose_transport(
                "RTP/AVP/TCP;unicast;interleaved=0-1,RTP/AVP;unicast;client_port=5000-5001"
            ),
            Some((5000, 5001))
        );
    }

    #[test]
    fn transport_quoted_mode_commas_do_not_split_specs() {
        // §12.39 grammar: mode = <"> 1#mode <"> — the comma inside the
        // quotes is part of one spec, not a spec separator.
        assert_eq!(
            choose_transport(r#"RTP/AVP;unicast;client_port=6000-6001;mode="PLAY,RECORD""#),
            Some((6000, 6001))
        );
        // A mode list without PLAY is not servable (§12.39: default PLAY;
        // v1 plays only).
        assert_eq!(
            choose_transport(r#"RTP/AVP;unicast;client_port=6000-6001;mode="RECORD""#),
            None
        );
    }

    #[test]
    fn transport_client_port_implies_unicast() {
        // §12.39 defaults delivery to multicast when neither word appears,
        // but client_port is defined as "the unicast RTP/RTCP port pair"
        // — a bare offer naming it is accepted as unicast.
        assert_eq!(
            choose_transport("RTP/AVP;client_port=7000-7001"),
            Some((7000, 7001))
        );
    }

    #[test]
    fn transport_without_client_port_is_rejected() {
        // Nowhere to send: not an acceptable unicast spec.
        assert_eq!(choose_transport("RTP/AVP;unicast"), None);
        assert_eq!(choose_transport(""), None);
    }

    #[test]
    fn transport_lone_client_port_expands_to_a_pair() {
        // §12.39 specifies a range; a lone port is read as p-(p+1) (RTP
        // even, RTCP above — RTP convention), matching the client's
        // tolerance.
        assert_eq!(choose_transport("RTP/AVP;unicast;client_port=5000"), Some((5000, 5001)));
        // ... unless p+1 would overflow the port space.
        assert_eq!(choose_transport("RTP/AVP;unicast;client_port=65535"), None);
    }

    #[test]
    fn transport_tokens_match_case_insensitively() {
        // Header field *values* reach us verbatim; tokens compare
        // case-insensitively like the client's Transport parser.
        assert_eq!(
            choose_transport("rtp/avp;UNICAST;Client_Port=4000-4001"),
            Some((4000, 4001))
        );
    }

    #[test]
    fn session_ids_are_long_and_distinct() {
        let peer: SocketAddr = "127.0.0.1:9".parse().unwrap();
        let a = generate_session_id(&peer);
        let b = generate_session_id(&peer);
        // §3.4: "MUST be at least eight octets long".
        assert!(a.len() >= 8, "got {a:?}");
        assert_ne!(a, b, "consecutive ids must differ");
    }

    #[test]
    fn session_header_id_strips_reflected_parameters() {
        assert_eq!(session_header_id("f00dcafe"), "f00dcafe");
        // Tolerated: a client reflecting the §12.37 response parameters.
        assert_eq!(session_header_id("f00dcafe;timeout=60"), "f00dcafe");
        assert_eq!(session_header_id(" f00dcafe ;x"), "f00dcafe");
    }

    #[test]
    fn request_heads_parse_and_reject_garbage() {
        let req = parse_head(
            "SETUP rtsp://h/s/trackID=0 RTSP/1.0\r\nCSeq: 2\r\nTransport: RTP/AVP;unicast;\r\n client_port=4588-4589\r\n",
        )
        .unwrap();
        assert_eq!(req.method, "SETUP");
        assert_eq!(req.uri, "rtsp://h/s/trackID=0");
        assert_eq!(req.version, "RTSP/1.0");
        assert_eq!(req.header("cseq"), Some("2"));
        // [H4.2] folded continuation line rejoins the field value.
        assert_eq!(
            req.header("Transport"),
            Some("RTP/AVP;unicast; client_port=4588-4589")
        );

        assert!(parse_head("NOT A VALID REQUEST\r\n").is_err()); // no RTSP/ version
        assert!(parse_head("BLARG\r\n").is_err()); // no URI/version at all
        assert!(parse_head("GET / HTTP/1.1\r\n").is_err()); // wrong protocol
        assert!(parse_head("OPTIONS * RTSP/1.0\r\nbroken header line\r\n").is_err());
    }
}
