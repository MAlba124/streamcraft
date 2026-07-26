//! A blocking RTSP 1.0 client (RFC 2326) over `std::net::TcpStream`.
//!
//! RTSP looks like HTTP/1.1 — text request/status lines, header fields, an
//! optional body framed by `Content-Length` (§4 defers message shape to
//! [H4]) — but differs where it matters to a client:
//!
//! - **CSeq** (§12.17) pairs every request with its response; both carry the
//!   same number. [`RtspClient`] numbers requests monotonically and checks
//!   the echo.
//! - **Session** (§12.37): the SETUP response names a server-chosen session
//!   (`Session: id[;timeout=seconds]`, timeout defaulting to 60) that the
//!   client "MUST return ... for any request related to that session".
//!   Tracked automatically once SETUP succeeds.
//! - **The connection carries no entity unless framed**: a response body
//!   exists only when `Content-Length` says so (§4.4; RTSP has no chunked
//!   coding).
//!
//! Method helpers cover the client half of §10: [`options`]
//! (§10.1), [`describe`] (§10.2, `Accept: application/sdp`, body parsed by
//! [`crate::sdp`]), [`setup`] (§10.4 with a §12.39 UDP-unicast Transport
//! offer), [`play`]/[`pause`] (§10.5/§10.6, aggregate — issued on the
//! presentation base URL), [`teardown`] (§10.7) and [`keepalive`]
//! (GET_PARAMETER with no body, §10.8: "may be used to test client or
//! server liveness ('ping')").
//!
//! **Auth** (RFC 2326 §12.5/§12.55 adopt HTTP's Authorization/
//! WWW-Authenticate; RFC 2617 defines them): on a 401 the client answers
//! Basic (§2) or Digest (§3.2, `qop="auth"` and the legacy RFC 2069 no-qop
//! form; MD5 from the in-tree `md5` module) and retries **once**; a second
//! 401 is
//! surfaced as [`RtspError::Unauthorized`]. Once challenged, subsequent
//! requests send `Authorization` preemptively with the same nonce
//! (RFC 2617 §3.3 authentication-session reuse).
//!
//! Interleaved (`RTP/AVP/TCP`) sessions are demuxed by
//! [`crate::interleaved`], not here: this client assumes only RTSP replies
//! arrive on the control connection (the UDP-transport case). For the TCP
//! case, drive the handshake here, then [`RtspClient::into_stream`] hands
//! the socket plus any unconsumed bytes to a demux loop.
//!
//! [`options`]: RtspClient::options
//! [`describe`]: RtspClient::describe
//! [`setup`]: RtspClient::setup
//! [`play`]: RtspClient::play
//! [`pause`]: RtspClient::pause
//! [`teardown`]: RtspClient::teardown
//! [`keepalive`]: RtspClient::keepalive

// Protocol-layer blocking sockets, sanctioned: the RTSP control connection
// runs on app/controller threads (spec: no-bins), never inside an element's
// scheduler group — the workspace element-IO lint does not apply here.
#![allow(clippy::disallowed_methods)]

use std::fmt;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use crate::md5::md5_hex;
use crate::sdp::{self, encode_base64, Sdp};

/// Cap on a response head, so a peer that never terminates its headers
/// can't make us buffer unboundedly (same guard as sc-http's).
const MAX_HEAD_BYTES: usize = 64 * 1024;

/// Cap on a response body — DESCRIBE SDP payloads are a few KB; anything
/// past this is a broken or hostile peer, not a session description.
const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

/// Default socket read/write timeout applied by [`RtspClient::connect`], so
/// a wedged server fails the session instead of blocking forever.
/// [`RtspClient::from_stream`] leaves the socket's own configuration alone.
const IO_TIMEOUT: Duration = Duration::from_secs(10);

/// Why an RTSP exchange failed.
#[derive(Debug)]
pub enum RtspError {
    /// Socket-level failure (connect, read, write, timeout).
    Io(std::io::Error),
    /// The peer's bytes do not parse as an RTSP message (§4), or violate
    /// request/response pairing (a CSeq echo mismatch, §12.17).
    Protocol(String),
    /// A non-2xx final status (§7.1.1 status codes follow HTTP's classes).
    Status(u16, String),
    /// 401 that could not be answered (no credentials, an unsupported
    /// challenge) or that persisted after the single retry.
    Unauthorized(String),
    /// The DESCRIBE body is not parseable SDP.
    Sdp(sdp::SdpError),
}

impl fmt::Display for RtspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RtspError::Io(e) => write!(f, "rtsp: io: {e}"),
            RtspError::Protocol(m) => write!(f, "rtsp: protocol: {m}"),
            RtspError::Status(code, reason) => write!(f, "rtsp: server returned {code} {reason}"),
            RtspError::Unauthorized(m) => write!(f, "rtsp: unauthorized: {m}"),
            RtspError::Sdp(e) => write!(f, "rtsp: describe body: {e}"),
        }
    }
}

impl std::error::Error for RtspError {}

impl From<std::io::Error> for RtspError {
    fn from(e: std::io::Error) -> Self {
        RtspError::Io(e)
    }
}

/// The server-chosen session (§12.37): `Session: id[;timeout=seconds]`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Session {
    pub id: String,
    /// Server inactivity timeout in seconds — how long it will "wait between
    /// RTSP commands before closing the session due to lack of activity";
    /// §12.37 defaults it to 60. Drive [`RtspClient::keepalive`] well inside
    /// this.
    pub timeout_secs: u64,
}

/// The server's `Transport` response header (§12.39), reduced to the
/// parameters a unicast UDP client acts on.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TransportInfo {
    /// `server_port=lo-hi`: the server's RTP/RTCP port pair.
    pub server_port: Option<(u16, u16)>,
    /// `client_port=lo-hi` echoed back (§12.39: the pair the client chose).
    pub client_port: Option<(u16, u16)>,
    /// `source=addr`: where the stream will come from, "if the source
    /// address for the stream is different than can be derived from the
    /// RTSP endpoint address" (§12.39).
    pub source: Option<String>,
    /// The full header value verbatim, for parameters not parsed here
    /// (`ssrc`, `destination`, `interleaved`, ...).
    pub raw: String,
}

/// A DESCRIBE result: the parsed SDP plus what SETUP needs around it.
#[derive(Clone, Debug)]
pub struct Described {
    pub sdp: Sdp,
    /// The response body verbatim (fixture capture, logging).
    pub raw: String,
    /// The presentation base URL for `a=control` resolution — §C.1.1's
    /// precedence: Content-Base, then Content-Location, then the request
    /// URL. Feed to [`resolve_control`].
    pub base: String,
}

/// One parsed RTSP response.
#[derive(Clone, Debug)]
pub struct Response {
    pub status: u16,
    pub reason: String,
    /// Header fields in arrival order (names case-insensitive, [H4.2]).
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    /// First value of header `name`, case-insensitively ([H4.2]).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Resolve an SDP `a=control` value against the presentation base URL
/// (RFC 2326 §C.1.1). The base comes from the DESCRIBE response
/// ([`Described::base`]); the rules:
///
/// - `*` (or an empty value) "is treated as if it were an empty embedded
///   URL, and thus inherits the entire base URL" (§C.1.1);
/// - an absolute URL (it has a `<scheme>://`) stands alone;
/// - an absolute path replaces the base's path;
/// - anything else is **appended** to the base path. Note this deviates from
///   strict RFC 1808 §4 (which would *replace* the last segment of a base
///   not ending in `/`): RTSP servers hand out track controls like
///   `trackID=0` expecting `<base>/trackID=0`, and bases regularly arrive
///   without the trailing slash — appending is the ecosystem-wide reading.
pub fn resolve_control(base: &str, control: &str) -> String {
    if control.is_empty() || control == "*" {
        return base.to_string();
    }
    if control.contains("://") {
        return control.to_string();
    }
    if let (Some(path), Some(scheme_end)) = (control.strip_prefix('/'), base.find("://")) {
        let auth_start = scheme_end + 3;
        let auth_end = base[auth_start..]
            .find('/')
            .map_or(base.len(), |i| auth_start + i);
        return format!("{}/{}", &base[..auth_end], path);
    }
    format!("{}/{}", base.trim_end_matches('/'), control)
}

/// Compute the Digest `response` value (RFC 2617 §3.2.2). Every hash input
/// is a colon-joined string of unquoted directive values (§3.2.2.4):
///
/// - `H(A1)` where `A1 = username ":" realm ":" password` (§3.2.2.2, the
///   default `MD5` algorithm),
/// - `H(A2)` where `A2 = Method ":" digest-uri` (§3.2.2.3, qop absent or
///   `auth` — `auth-int` would hash the body too and is not supported),
/// - with `qop=auth` (§3.2.2.1):
///   `response = H( H(A1) ":" nonce ":" nc ":" cnonce ":" "auth" ":" H(A2) )`,
/// - without qop (the RFC 2069 compatibility form, §3.2.2.1):
///   `response = H( H(A1) ":" nonce ":" H(A2) )`.
pub fn digest_response(
    username: &str,
    realm: &str,
    password: &str,
    method: &str,
    uri: &str,
    nonce: &str,
    qop_auth: Option<(&str, &str)>,
) -> String {
    let ha1 = md5_hex(format!("{username}:{realm}:{password}").as_bytes());
    let ha2 = md5_hex(format!("{method}:{uri}").as_bytes());
    match qop_auth {
        Some((nc, cnonce)) => md5_hex(format!("{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}").as_bytes()),
        None => md5_hex(format!("{ha1}:{nonce}:{ha2}").as_bytes()),
    }
}

/// A parsed WWW-Authenticate challenge we can answer (RFC 2617 §1.2).
#[derive(Clone, Debug)]
enum AuthState {
    /// §2: `Authorization: Basic base64(userid ":" password)`.
    Basic,
    /// §3.2.1 challenge directives we replay in each Authorization header.
    Digest {
        realm: String,
        nonce: String,
        /// §3.2.1: opaque is returned "unchanged in the Authorization
        /// header of subsequent requests".
        opaque: Option<String>,
        /// The server offered `qop="...,auth,..."` — we then send the
        /// §3.2.2 `qop=auth`, `nc`, `cnonce` triple.
        qop_auth: bool,
        /// nonce-count of the *next* request (§3.2.2: "the count of the
        /// number of requests ... that the client has sent with the nonce
        /// value in this request", hex, starting 00000001).
        nc: u32,
    },
}

/// A blocking RTSP 1.0 client on one control connection.
pub struct RtspClient {
    stream: TcpStream,
    /// The presentation (request) URL — also §C.1.1's fallback base.
    url: String,
    /// Next request's CSeq (§12.17). Starts at 1.
    cseq: u32,
    session: Option<Session>,
    /// Presentation base recorded from DESCRIBE (§C.1.1 rules 1–2).
    base: Option<String>,
    creds: Option<(String, String)>,
    auth: Option<AuthState>,
    /// Bytes read past the previous response (a peer may batch writes).
    inbuf: Vec<u8>,
    /// Fixed client nonce for deterministic tests; generated otherwise.
    cnonce: Option<String>,
}

impl RtspClient {
    /// Connect to the server named by an `rtsp://host[:port]/...` URL
    /// (§3.2: "If the port is empty or not given, port 554 is assumed").
    /// Applies a default 10 s read/write timeout to the socket.
    pub fn connect(url: &str) -> Result<Self, RtspError> {
        let (host, port) = parse_rtsp_url(url)?;
        let stream = TcpStream::connect((host.as_str(), port))?;
        stream.set_read_timeout(Some(IO_TIMEOUT))?;
        stream.set_write_timeout(Some(IO_TIMEOUT))?;
        Ok(Self::from_stream(stream, url))
    }

    /// Wrap a pre-connected control connection. `url` is the presentation
    /// URL used on request lines (and as the §C.1.1 fallback base). The
    /// socket's timeouts are left as configured by the caller.
    pub fn from_stream(stream: TcpStream, url: impl Into<String>) -> Self {
        Self {
            stream,
            url: url.into(),
            cseq: 1,
            session: None,
            base: None,
            creds: None,
            auth: None,
            inbuf: Vec::new(),
            cnonce: None,
        }
    }

    /// Credentials for answering 401 challenges (RFC 2617). Without them a
    /// 401 surfaces as [`RtspError::Unauthorized`].
    pub fn set_credentials(&mut self, username: impl Into<String>, password: impl Into<String>) {
        self.creds = Some((username.into(), password.into()));
    }

    /// Pin the Digest client nonce (§3.2.2 `cnonce`) instead of deriving one
    /// from the clock — for deterministic tests.
    pub fn set_cnonce(&mut self, cnonce: impl Into<String>) {
        self.cnonce = Some(cnonce.into());
    }

    /// The session established by SETUP, until TEARDOWN (§12.37).
    pub fn session(&self) -> Option<&Session> {
        self.session.as_ref()
    }

    /// The presentation base URL: Content-Base/Content-Location from
    /// DESCRIBE when seen, else the request URL (§C.1.1's order).
    pub fn base_url(&self) -> &str {
        self.base.as_deref().unwrap_or(&self.url)
    }

    /// The underlying control connection (e.g. to adjust timeouts).
    pub fn stream(&self) -> &TcpStream {
        &self.stream
    }

    /// Dismantle into the socket plus any already-read, unconsumed bytes —
    /// the handoff for an interleaved-TCP receive loop feeding
    /// [`crate::interleaved::Demux`].
    pub fn into_stream(self) -> (TcpStream, Vec<u8>) {
        (self.stream, self.inbuf)
    }

    // --- methods (§10) ------------------------------------------------------

    /// OPTIONS (§10.1) on the presentation URL. Returns the methods the
    /// server advertises in `Public` (the §10.1 example's reply header).
    pub fn options(&mut self) -> Result<Vec<String>, RtspError> {
        let url = self.url.clone();
        let resp = self.request("OPTIONS", &url, &[], &[])?;
        let resp = ok(resp)?;
        Ok(resp
            .header("Public")
            .map(|v| v.split(',').map(|m| m.trim().to_string()).collect())
            .unwrap_or_default())
    }

    /// DESCRIBE (§10.2) with `Accept: application/sdp`. Parses the SDP body
    /// and records the presentation base (§C.1.1: Content-Base, then
    /// Content-Location, then the request URL) for later [`Self::base_url`]
    /// aggregate operations.
    pub fn describe(&mut self) -> Result<Described, RtspError> {
        let url = self.url.clone();
        let resp = self.request("DESCRIBE", &url, &[("Accept", "application/sdp")], &[])?;
        let resp = ok(resp)?;

        // §10.2: we asked for application/sdp; a different Content-Type is a
        // description we can't read — fail loudly rather than misparse.
        if let Some(ct) = resp.header("Content-Type") {
            if !ct.trim().starts_with("application/sdp") {
                return Err(RtspError::Protocol(format!(
                    "DESCRIBE returned Content-Type {ct:?}, wanted application/sdp"
                )));
            }
        }

        let base = resp
            .header("Content-Base")
            .or_else(|| resp.header("Content-Location"))
            .unwrap_or(self.url.as_str())
            .to_string();
        let raw = String::from_utf8_lossy(&resp.body).into_owned();
        let sdp = sdp::parse(&raw).map_err(RtspError::Sdp)?;
        self.base = Some(base.clone());
        Ok(Described { sdp, raw, base })
    }

    /// SETUP (§10.4) one media stream at `control_url` (resolve it first via
    /// [`resolve_control`]), offering UDP unicast delivery to
    /// `client_port=rtp_port-(rtp_port+1)` (§12.39; RTP on the even port,
    /// RTCP one above, per RTP convention). Records the Session the response
    /// names (§12.37) and returns the server's Transport parameters.
    pub fn setup(&mut self, control_url: &str, rtp_port: u16) -> Result<TransportInfo, RtspError> {
        let transport = format!("RTP/AVP;unicast;client_port={}-{}", rtp_port, rtp_port + 1);
        let resp = self.request("SETUP", control_url, &[("Transport", &transport)], &[])?;
        let resp = ok(resp)?;

        // §12.37: "Once a client receives a Session identifier, it MUST
        // return it for any request related to that session."
        match resp.header("Session") {
            Some(v) => self.session = Some(parse_session(v)),
            None if self.session.is_none() => {
                return Err(RtspError::Protocol(
                    "SETUP response carried no Session header (§12.37)".into(),
                ));
            }
            None => {} // later SETUPs may rely on the established session
        }

        let raw = resp.header("Transport").unwrap_or_default();
        Ok(parse_transport(raw))
    }

    /// PLAY (§10.5), aggregate: issued on the presentation base URL so every
    /// SETUP-up stream starts. No Range header — "starts playing a stream
    /// from the beginning unless the stream has been paused" (§10.5).
    pub fn play(&mut self) -> Result<(), RtspError> {
        let url = self.base_url().to_string();
        let resp = self.request("PLAY", &url, &[], &[])?;
        ok(resp).map(drop)
    }

    /// PAUSE (§10.6), aggregate: "temporarily halts ... delivery, without
    /// freeing server resources"; a later PLAY resumes.
    pub fn pause(&mut self) -> Result<(), RtspError> {
        let url = self.base_url().to_string();
        let resp = self.request("PAUSE", &url, &[], &[])?;
        ok(resp).map(drop)
    }

    /// TEARDOWN (§10.7): stops delivery and frees the session; the tracked
    /// Session is forgotten on success.
    pub fn teardown(&mut self) -> Result<(), RtspError> {
        let url = self.base_url().to_string();
        let resp = self.request("TEARDOWN", &url, &[], &[])?;
        ok(resp)?;
        self.session = None;
        Ok(())
    }

    /// Session keepalive: GET_PARAMETER with no entity body — §10.8: it
    /// "may be used to test client or server liveness ('ping')". Send well
    /// inside [`Session::timeout_secs`].
    pub fn keepalive(&mut self) -> Result<(), RtspError> {
        let url = self.base_url().to_string();
        let resp = self.request("GET_PARAMETER", &url, &[], &[])?;
        ok(resp).map(drop)
    }

    // --- transport ----------------------------------------------------------

    /// Send one request and read its response, transparently answering a
    /// single 401 challenge (RFC 2617 §1.2) when credentials are set. The
    /// final response is returned whatever its status — helpers layer 2xx
    /// checks on top — except that a 401 *after* the retry (or one we cannot
    /// answer while holding credentials) becomes
    /// [`RtspError::Unauthorized`].
    pub fn request(
        &mut self,
        method: &str,
        url: &str,
        extra_headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Response, RtspError> {
        let resp = self.round_trip(method, url, extra_headers, body)?;
        if resp.status != 401 || self.creds.is_none() {
            return Ok(resp);
        }
        // Answer the newest challenge and retry exactly once (RFC 2617
        // §1.2's challenge→response shape). Re-parsing on every 401 also
        // covers `stale=true` nonce rotation (§3.2.1): the retry always
        // uses the freshest nonce.
        self.auth = Some(pick_challenge(&resp)?);
        let retry = self.round_trip(method, url, extra_headers, body)?;
        if retry.status == 401 {
            return Err(RtspError::Unauthorized(
                "credentials refused (401 after retry)".into(),
            ));
        }
        Ok(retry)
    }

    /// One request/response exchange: CSeq stamping (§12.17), Session echo
    /// (§12.37), preemptive Authorization when a challenge was accepted.
    fn round_trip(
        &mut self,
        method: &str,
        url: &str,
        extra_headers: &[(&str, &str)],
        body: &[u8],
    ) -> Result<Response, RtspError> {
        let cseq = self.cseq;
        self.cseq += 1;

        // §6.1 Request-Line: Method SP Request-URI SP RTSP-Version CRLF.
        let mut req = format!("{method} {url} RTSP/1.0\r\nCSeq: {cseq}\r\n");
        req.push_str("User-Agent: streamcraft\r\n");
        if let Some(s) = &self.session {
            req.push_str(&format!("Session: {}\r\n", s.id));
        }
        if let Some(a) = self.authorization(method, url) {
            req.push_str(&format!("Authorization: {a}\r\n"));
        }
        for (k, v) in extra_headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        if !body.is_empty() {
            req.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        req.push_str("\r\n");

        self.stream.write_all(req.as_bytes())?;
        if !body.is_empty() {
            self.stream.write_all(body)?;
        }

        let resp = self.read_response()?;

        // §12.17: "For every RTSP request containing the given sequence
        // number, there will be a corresponding response having the same
        // number." A different echo means we're reading someone else's
        // reply. (A missing CSeq is tolerated: nothing to cross-check.)
        if let Some(echo) = resp.header("CSeq") {
            if echo.trim().parse::<u32>() != Ok(cseq) {
                return Err(RtspError::Protocol(format!(
                    "CSeq mismatch: sent {cseq}, response says {echo:?}"
                )));
            }
        }
        Ok(resp)
    }

    /// The Authorization header value for the current auth state, if any.
    /// Digest bumps its nonce-count per use (§3.2.2 `nc`).
    fn authorization(&mut self, method: &str, uri: &str) -> Option<String> {
        let (user, pass) = self.creds.clone()?;
        match self.auth.as_mut()? {
            // §2: base64 of "userid:password".
            AuthState::Basic => Some(format!(
                "Basic {}",
                encode_base64(format!("{user}:{pass}").as_bytes())
            )),
            AuthState::Digest {
                realm,
                nonce,
                opaque,
                qop_auth,
                nc,
            } => {
                let mut h = format!("Digest username=\"{user}\", realm=\"{realm}\", nonce=\"{nonce}\", uri=\"{uri}\"");
                if *qop_auth {
                    // §3.2.2: nc is 8 lowercase hex digits; cnonce is an
                    // opaque client-chosen string.
                    let nc_str = format!("{:08x}", *nc);
                    *nc += 1;
                    let cnonce = self.cnonce.clone().unwrap_or_else(gen_cnonce);
                    let response = digest_response(
                        &user,
                        realm,
                        &pass,
                        method,
                        uri,
                        nonce,
                        Some((&nc_str, &cnonce)),
                    );
                    // §3.2.2 grammar: message-qop and nonce-count values are
                    // tokens (unquoted); cnonce and response are quoted.
                    h.push_str(&format!(
                        ", response=\"{response}\", qop=auth, nc={nc_str}, cnonce=\"{cnonce}\""
                    ));
                } else {
                    let response = digest_response(&user, realm, &pass, method, uri, nonce, None);
                    h.push_str(&format!(", response=\"{response}\""));
                }
                if let Some(o) = opaque {
                    // §3.2.1: opaque is returned unchanged.
                    h.push_str(&format!(", opaque=\"{o}\""));
                }
                Some(h)
            }
        }
    }

    /// Read one RTSP response: status line + headers to the blank line, then
    /// a `Content-Length` body if declared (§4.4; absent means none).
    fn read_response(&mut self) -> Result<Response, RtspError> {
        // Accumulate until the head terminator.
        let head_len = loop {
            // Robustness: drop stray blank lines between responses (they
            // would otherwise read as an instant empty head).
            let lead = self
                .inbuf
                .iter()
                .take_while(|&&b| b == b'\r' || b == b'\n')
                .count();
            if lead > 0 {
                self.inbuf.drain(..lead);
            }
            if let Some(n) = head_end(&self.inbuf) {
                break n;
            }
            if self.inbuf.len() > MAX_HEAD_BYTES {
                return Err(RtspError::Protocol(format!(
                    "response head exceeded {MAX_HEAD_BYTES} bytes"
                )));
            }
            self.fill()?;
        };

        let head = String::from_utf8_lossy(&self.inbuf[..head_len]).into_owned();
        let mut lines = head.lines();
        let status_line = lines.next().unwrap_or("");

        // §7.1 Status-Line: RTSP-Version SP Status-Code SP Reason-Phrase.
        let mut parts = status_line.splitn(3, ' ');
        let version = parts.next().unwrap_or("");
        if !version.starts_with("RTSP/") {
            return Err(RtspError::Protocol(format!(
                "malformed status line {status_line:?}"
            )));
        }
        let status = parts
            .next()
            .and_then(|c| c.parse::<u16>().ok())
            .ok_or_else(|| RtspError::Protocol(format!("malformed status line {status_line:?}")))?;
        let reason = parts.next().unwrap_or("").trim().to_string();

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
                return Err(RtspError::Protocol(format!(
                    "malformed header line {line:?}"
                )));
            };
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }

        // Body framing (§4.4): exactly Content-Length octets, or none.
        let body_len = match headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("Content-Length"))
        {
            Some((_, v)) => v
                .trim()
                .parse::<usize>()
                .map_err(|_| RtspError::Protocol(format!("unparsable Content-Length {v:?}")))?,
            None => 0,
        };
        if body_len > MAX_BODY_BYTES {
            return Err(RtspError::Protocol(format!(
                "Content-Length {body_len} exceeds the {MAX_BODY_BYTES} sanity cap"
            )));
        }
        while self.inbuf.len() < head_len + body_len {
            self.fill()?;
        }
        let body = self.inbuf[head_len..head_len + body_len].to_vec();
        self.inbuf.drain(..head_len + body_len);

        Ok(Response {
            status,
            reason,
            headers,
            body,
        })
    }

    /// Read some more socket bytes into `inbuf`; EOF is a protocol error
    /// mid-response (the caller only asks when more bytes are owed).
    fn fill(&mut self) -> Result<(), RtspError> {
        let mut chunk = [0u8; 4096];
        let n = self.stream.read(&mut chunk)?;
        if n == 0 {
            return Err(RtspError::Protocol("connection closed mid-response".into()));
        }
        self.inbuf.extend_from_slice(&chunk[..n]);
        Ok(())
    }
}

/// Map a final response to the 2xx-only contract of the method helpers.
fn ok(resp: Response) -> Result<Response, RtspError> {
    match resp.status {
        200..=299 => Ok(resp),
        401 => Err(RtspError::Unauthorized(
            "server requires authentication (set_credentials)".into(),
        )),
        _ => Err(RtspError::Status(resp.status, resp.reason)),
    }
}

/// Pick the strongest answerable challenge from the 401's WWW-Authenticate
/// headers: Digest over Basic (RFC 2617 §4.6: use "the strongest auth-scheme
/// it understands"). Each header is treated as one challenge (scheme token
/// first) — servers offering both send separate headers in practice.
fn pick_challenge(resp: &Response) -> Result<AuthState, RtspError> {
    let mut basic = false;
    for (k, v) in &resp.headers {
        if !k.eq_ignore_ascii_case("WWW-Authenticate") {
            continue;
        }
        let v = v.trim();
        let (scheme, params) = match v.split_once(char::is_whitespace) {
            Some((s, p)) => (s, p),
            None => (v, ""),
        };
        if scheme.eq_ignore_ascii_case("Basic") {
            basic = true;
        } else if scheme.eq_ignore_ascii_case("Digest") {
            let params = parse_auth_params(params);
            let get = |n: &str| params.iter().find(|(k, _)| k == n).map(|(_, v)| v.clone());
            // §3.2.1: algorithm defaults to MD5; MD5-sess (and anything
            // else) is not implemented — fall through to other challenges.
            if let Some(alg) = get("algorithm") {
                if !alg.eq_ignore_ascii_case("MD5") {
                    continue;
                }
            }
            // §3.2.1 qop-options: if offered, we must pick one we support.
            let qop_auth = match get("qop") {
                Some(q) => {
                    if !q.split(',').any(|o| o.trim().eq_ignore_ascii_case("auth")) {
                        continue; // e.g. qop="auth-int" only — unanswerable
                    }
                    true
                }
                None => false, // legacy RFC 2069 form
            };
            let (Some(realm), Some(nonce)) = (get("realm"), get("nonce")) else {
                continue; // §3.2.1 makes both mandatory
            };
            return Ok(AuthState::Digest {
                realm,
                nonce,
                opaque: get("opaque"),
                qop_auth,
                nc: 1,
            });
        }
    }
    if basic {
        return Ok(AuthState::Basic);
    }
    Err(RtspError::Unauthorized(
        "401 without an answerable WWW-Authenticate challenge".into(),
    ))
}

/// Split an auth-param list (RFC 2617 §1.2: comma-separated
/// `name=value` with values as tokens or quoted-strings). Quoted values may
/// contain commas and `\"` quoted-pairs. Names are lowercased.
fn parse_auth_params(s: &str) -> Vec<(String, String)> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        while i < b.len() && (b[i] == b' ' || b[i] == b'\t' || b[i] == b',') {
            i += 1;
        }
        let name_start = i;
        while i < b.len() && b[i] != b'=' && b[i] != b',' {
            i += 1;
        }
        let name = s[name_start..i].trim().to_ascii_lowercase();
        if i >= b.len() || b[i] == b',' {
            continue; // a bare token (no value) — nothing we use
        }
        i += 1; // '='
        let value = if i < b.len() && b[i] == b'"' {
            i += 1;
            let mut v = Vec::new();
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' && i + 1 < b.len() {
                    i += 1; // quoted-pair: keep the escaped byte
                }
                v.push(b[i]);
                i += 1;
            }
            i += 1; // closing '"'
            String::from_utf8_lossy(&v).into_owned()
        } else {
            let vs = i;
            while i < b.len() && b[i] != b',' {
                i += 1;
            }
            s[vs..i].trim().to_string()
        };
        if !name.is_empty() {
            out.push((name, value));
        }
    }
    out
}

/// A clock-derived cnonce (§3.2.2). The cnonce only salts the digest against
/// chosen-plaintext on the server nonce — uniqueness matters, secrecy does
/// not, so the wall clock suffices and keeps the crate dependency-free.
fn gen_cnonce() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}{:08x}", now.as_secs(), now.subsec_nanos())
}

/// Parse a `Session` response header (§12.37):
/// `session-id [ ";" "timeout" "=" delta-seconds ]`, default timeout 60 s.
fn parse_session(value: &str) -> Session {
    let mut parts = value.split(';');
    let id = parts.next().unwrap_or("").trim().to_string();
    let mut timeout_secs = 60;
    for p in parts {
        if let Some(t) = p.trim().strip_prefix("timeout=") {
            if let Ok(t) = t.trim().parse::<u64>() {
                timeout_secs = t;
            }
        }
    }
    Session { id, timeout_secs }
}

/// Parse a `Transport` response header (§12.39): the response carries a
/// single transport-spec (the server "MUST return a single option") of
/// semicolon-separated parameters; a comma would start another spec, so
/// anything past one is ignored defensively.
fn parse_transport(value: &str) -> TransportInfo {
    let mut info = TransportInfo {
        server_port: None,
        client_port: None,
        source: None,
        raw: value.to_string(),
    };
    let spec = value.split(',').next().unwrap_or("");
    for param in spec.split(';') {
        let Some((k, v)) = param.trim().split_once('=') else {
            continue; // unicast / RTP/AVP / plain flags
        };
        let v = v.trim();
        if k.eq_ignore_ascii_case("server_port") {
            info.server_port = parse_port_pair(v);
        } else if k.eq_ignore_ascii_case("client_port") {
            info.client_port = parse_port_pair(v);
        } else if k.eq_ignore_ascii_case("source") {
            info.source = Some(v.to_string());
        }
    }
    info
}

/// `lo-hi` port ranges (§12.39: "It is specified as a range, e.g.,
/// client_port=3456-3457"); a lone port is taken as `p-(p+1)`.
fn parse_port_pair(v: &str) -> Option<(u16, u16)> {
    match v.split_once('-') {
        Some((lo, hi)) => Some((lo.trim().parse().ok()?, hi.trim().parse().ok()?)),
        None => {
            let p: u16 = v.trim().parse().ok()?;
            Some((p, p.checked_add(1)?))
        }
    }
}

/// Index just past the response head's terminating blank line. Lines end
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

/// Host and port from an `rtsp://` URL (§3.2; default port 554). Bracketed
/// IPv6 literals and (ignored) userinfo are handled; the path stays in the
/// URL used on request lines.
fn parse_rtsp_url(url: &str) -> Result<(String, u16), RtspError> {
    let rest = url
        .strip_prefix("rtsp://")
        .ok_or_else(|| RtspError::Protocol(format!("only rtsp:// URLs are supported: {url}")))?;
    let authority = &rest[..rest.find('/').unwrap_or(rest.len())];
    let hostport = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);

    let (host, port_str) = if let Some(rest) = hostport.strip_prefix('[') {
        // RFC 2732 bracketed IPv6 literal: [addr][:port].
        let end = rest
            .find(']')
            .ok_or_else(|| RtspError::Protocol(format!("unterminated IPv6 literal in {url}")))?;
        (&rest[..end], rest[end + 1..].strip_prefix(':'))
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (hostport, None),
        }
    };
    if host.is_empty() {
        return Err(RtspError::Protocol(format!("missing host in {url}")));
    }
    let port = match port_str {
        Some(p) => p
            .parse::<u16>()
            .map_err(|_| RtspError::Protocol(format!("invalid port in {url}")))?,
        None => 554, // §3.2
    };
    Ok((host.to_string(), port))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RFC 2617 §3.5's worked example, end to end through our hash chain:
    /// username "Mufasa", realm "testrealm@host.com", password
    /// "Circle Of Life", method GET, uri "/dir/index.html", nonce
    /// "dcd98b7102dd2f0e8b11d0f600bfb0c093", qop=auth, nc=00000001,
    /// cnonce="0a4f113b" → response "6629fae49393a05397450978507c4ef1".
    #[test]
    fn digest_response_matches_rfc2617_example() {
        assert_eq!(
            digest_response(
                "Mufasa",
                "testrealm@host.com",
                "Circle Of Life",
                "GET",
                "/dir/index.html",
                "dcd98b7102dd2f0e8b11d0f600bfb0c093",
                Some(("00000001", "0a4f113b")),
            ),
            "6629fae49393a05397450978507c4ef1"
        );
    }

    /// §3.2.2.4's A1 illustration doubles as a no-qop check: HA1 =
    /// H("Mufasa:myhost@testrealm.com:Circle Of Life") feeds the RFC 2069
    /// form H(HA1 ":" nonce ":" HA2) — pin the composed output.
    #[test]
    fn digest_response_no_qop_form_is_stable() {
        let got = digest_response(
            "Mufasa",
            "myhost@testrealm.com",
            "Circle Of Life",
            "DESCRIBE",
            "rtsp://example.com/stream",
            "abc",
            None,
        );
        // Recomputed with the same md5 primitive — guards the string
        // assembly (colon placement, §3.2.2.1) rather than the hash itself.
        let ha1 = md5_hex(b"Mufasa:myhost@testrealm.com:Circle Of Life");
        let ha2 = md5_hex(b"DESCRIBE:rtsp://example.com/stream");
        assert_eq!(got, md5_hex(format!("{ha1}:abc:{ha2}").as_bytes()));
    }

    #[test]
    fn control_resolution_follows_c11() {
        // Absolute control URL wins.
        assert_eq!(
            resolve_control("rtsp://h/base/", "rtsp://other/x"),
            "rtsp://other/x"
        );
        // "*" inherits the entire base (§C.1.1).
        assert_eq!(resolve_control("rtsp://h/base/", "*"), "rtsp://h/base/");
        // Relative appends, with and without the base's trailing slash.
        assert_eq!(
            resolve_control("rtsp://h/base/", "trackID=1"),
            "rtsp://h/base/trackID=1"
        );
        assert_eq!(
            resolve_control("rtsp://h/base", "trackID=1"),
            "rtsp://h/base/trackID=1"
        );
        // Absolute path replaces the base path.
        assert_eq!(
            resolve_control("rtsp://h:8554/base/sub", "/root"),
            "rtsp://h:8554/root"
        );
    }

    #[test]
    fn session_header_parses_with_and_without_timeout() {
        let s = parse_session("12345678;timeout=30");
        assert_eq!(s.id, "12345678");
        assert_eq!(s.timeout_secs, 30);
        // §12.37: default timeout 60 s.
        let s = parse_session("f00dcafe");
        assert_eq!(s.id, "f00dcafe");
        assert_eq!(s.timeout_secs, 60);
    }

    #[test]
    fn transport_header_parses_ports_and_source() {
        let t = parse_transport(
            "RTP/AVP;unicast;client_port=5000-5001;server_port=6256-6257;\
             source=192.0.2.5;ssrc=DEADBEEF",
        );
        assert_eq!(t.client_port, Some((5000, 5001)));
        assert_eq!(t.server_port, Some((6256, 6257)));
        assert_eq!(t.source.as_deref(), Some("192.0.2.5"));
        assert!(t.raw.contains("ssrc=DEADBEEF"));
    }

    #[test]
    fn auth_params_handle_quotes_and_embedded_commas() {
        let p = parse_auth_params(
            r#"realm="a, realm", nonce="n1", qop="auth,auth-int", stale=FALSE, opaque="o\"x""#,
        );
        let get = |n: &str| p.iter().find(|(k, _)| k == n).map(|(_, v)| v.as_str());
        assert_eq!(get("realm"), Some("a, realm"));
        assert_eq!(get("nonce"), Some("n1"));
        assert_eq!(get("qop"), Some("auth,auth-int"));
        assert_eq!(get("stale"), Some("FALSE"));
        assert_eq!(get("opaque"), Some("o\"x"));
    }

    #[test]
    fn rtsp_urls_parse_host_and_default_port() {
        assert_eq!(
            parse_rtsp_url("rtsp://media.example.com/twister").unwrap(),
            ("media.example.com".to_string(), 554) // §3.2 default
        );
        assert_eq!(
            parse_rtsp_url("rtsp://10.0.0.2:8554/live/stream").unwrap(),
            ("10.0.0.2".to_string(), 8554)
        );
        assert_eq!(
            parse_rtsp_url("rtsp://user:pw@host:1554/x").unwrap(),
            ("host".to_string(), 1554)
        );
        assert_eq!(
            parse_rtsp_url("rtsp://[2001:db8::1]:8554/x").unwrap(),
            ("2001:db8::1".to_string(), 8554)
        );
        assert!(parse_rtsp_url("http://host/x").is_err());
        assert!(parse_rtsp_url("rtsp://host:99999/x").is_err());
    }

    #[test]
    fn head_end_accepts_crlf_and_bare_lf() {
        // "RTSP/1.0 200 OK\r\n" is 17 bytes, "CSeq: 1\r\n" is 9, the blank
        // line 2 → the head ends at 28, where BODY begins.
        assert_eq!(
            head_end(b"RTSP/1.0 200 OK\r\nCSeq: 1\r\n\r\nBODY"),
            Some(28)
        );
        assert_eq!(head_end(b"RTSP/1.0 200 OK\nCSeq: 1\n\nBODY"), Some(25));
        assert_eq!(head_end(b"RTSP/1.0 200 OK\r\nCSeq: 1\r\n"), None);
    }
}
