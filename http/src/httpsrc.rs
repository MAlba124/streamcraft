//! `httpsrc` — downloads a file over plain HTTP into pooled buffers
//! (spec: Milestone applications §2).
//!
//! `start()` opens a [`TcpStream`], sends a `GET`, parses the status line + headers,
//! and picks a **message-body framing mode** from those headers per RFC 9112 (checked
//! in at `spec/rfc9112.txt`). The connect + request write + header read are one-time
//! setup and stay synchronous. It then **hands the socket fd to the reactor** (spec:
//! IO): the connected `TcpStream` is consumed into a `File` (`into_raw_fd` →
//! `File::from_raw_fd`, so the fd has exactly one owner) and registered with
//! `ctx.io()`, exactly like `filesrc` registers its file.
//!
//! `process()` then drains completed body reads — submitted as streaming
//! [`OpKind::Recv`](profluens_core::io::OpKind::Recv) ops — decodes them into pooled
//! buffers and pushes them downstream, ending with [`Flow::Eos`] once the body is
//! complete. The socket ride is **single-op**: unlike `filesrc`'s positioned reads
//! (order-independent, pipelined `credits` deep), a streaming socket read consumes the
//! *next* bytes off the fd, so two in flight could reorder the byte stream. We keep at
//! most one `Recv` outstanding — sequential by nature — while still being fully async:
//! under the io_uring reactor the read never blocks the group thread, and completions
//! are drained across `process()` passes.
//!
//! ## Body framing (RFC 9112 §6.3 "Message Body Length")
//! The framing mode is chosen from the response headers, in order of precedence:
//! 1. **`Transfer-Encoding: chunked`** (RFC 9112 §6.1, §7.1) — overrides
//!    `Content-Length` (§6.3 rule 3). The chunk framing is decoded by a small state
//!    machine and the *decoded* bytes are emitted; the download ends after the zero
//!    (last) chunk.
//! 2. **`Content-Length: N`** (RFC 9112 §6.2) — exactly `N` body octets are emitted,
//!    then [`Flow::Eos`], without waiting for the socket to close.
//! 3. **Neither** — read to EOF (`Connection: close`, §6.3 rule 8). This is why the
//!    request sends `Connection: close`.
//!
//! Only the `chunked` transfer coding is decoded; any other transfer coding (e.g.
//! `gzip`) is rejected, as this element does not implement content decoders. For the
//! same reason the request asks for `Accept-Encoding: identity` and a response that
//! applies a `Content-Encoding` anyway is rejected (RFC 9110 §8.4): a content coding
//! survives de-chunking, so emitting it would hand coded bytes downstream as if they
//! were the resource.
//!
//! ## `https://` (TLS)
//! TLS rides the same architecture (see [`crate::tls`]): the handshake happens in
//! `start()` alongside the connect + header read, and afterwards the
//! [`rustls::ClientConnection`] stays behind as a **sans-IO decrypt stage** — the
//! reactor's `Recv` completions carry *ciphertext*, which `process()` feeds through
//! the state machine into `inbuf`; from there the framing decoder is oblivious to
//! the transport. End-of-body over TLS is authenticated: a close-delimited (EOF)
//! body requires the peer's `close_notify` (RFC 8446 §6.1) — a bare TCP FIN is an
//! attacker-forgeable truncation and errors. Length/chunk-framed bodies terminate
//! on their own framing, so a missing `close_notify` after a complete body is
//! tolerated (as common servers omit it).
//!
//! ## Redirects (RFC 9110 §15.4)
//! `start()` follows `301`, `302`, `303`, `307` and `308`, up to
//! [`MAX_REDIRECTS`] hops. The method stays `GET` throughout — including `303`
//! (§15.4.4), whose whole point is "retrieve the other resource **with GET**", which
//! is what we were already doing. Each hop is a fresh connect + TLS handshake, since
//! the `Location` may move host, port *and* scheme. The `Location` is resolved against
//! the request URI per RFC 3986 §5 ([`resolve_reference`]) — servers send absolute
//! URIs, absolute paths and (against §10.2.2's "SHOULD") bare relative paths alike.
//! The resolved URL becomes the **effective URL**: reconnects and ranged requests go
//! there, not back through the redirector.
//!
//! ## Ranges, seeking and resume (RFC 9110 §14)
//! The first response's `Accept-Ranges: bytes` (§14.3) and `Content-Length` are
//! recorded. A stream is seekable when ranges are supported *and* the total length is
//! known — the app reads both off [`HttpInfo`] (see [`HttpSrc::info`]) to build the
//! byte↔time mapping a proportional seek needs. On a seek the connection is torn down
//! and reopened with `Range: bytes=<target>-` (§14.2); the answer must be `206` and
//! its `Content-Range` (§14.4) must start exactly at the target, or the seek fails
//! loudly rather than silently handing back the wrong bytes.
//!
//! The same machinery is the **resume** path: a connection that dies mid-body (socket
//! error, or EOF with `Content-Length` bytes still owed) reconnects at the last byte
//! actually delivered downstream, with exponential backoff and a bounded number of
//! consecutive attempts. Resume requests carry `If-Range` with the first response's
//! strong `ETag` when there is one (§14.5), so a representation that changed under us
//! comes back as a `200` we refuse instead of a splice of two different files.
//!
//! ## Stall protection
//! A server that accepts the connection and then goes silent used to hang the group
//! forever: the reactor has no timeout op, and `Recv` on a blocking fd waits without
//! end. Every socket therefore gets `SO_RCVTIMEO` (and `SO_SNDTIMEO`) before it is
//! registered — the same `set_read_timeout` guard `rtsp/src/client.rs` puts on its
//! control connection — so dead air surfaces as a `WouldBlock`/`TimedOut` completion
//! that feeds the reconnect path. Under an async backend (io_uring) the socket option
//! does not apply to submitted reads; a reactor-level timeout op is the real fix and
//! is future work, noted at [`HttpSrc::stall_timeout`].
//!
//! ## v1 limitations
//! - **No client writes after the request**: a server-initiated TLS 1.3 KeyUpdate
//!   or rekey mid-download gets its response queued but never flushed (the reactor
//!   has no streaming send yet); receive-direction decryption is unaffected.
//! - **Connect, handshake and header read block the group thread**, on the first
//!   connection and on every reconnect (see [`HttpSrc::open`]).

use std::borrow::Cow;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::unix::io::{FromRawFd, IntoRawFd};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::tls;
use profluens_core::batch::Inputs;
use profluens_core::buffer::BufferFlags;
use profluens_core::bus::BusMessage;
use profluens_core::ctx::Ctx;
use profluens_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use profluens_core::error::Error;
use profluens_core::event::Event;
use profluens_core::format::OfferDesc;
use profluens_core::id::PadId;
use profluens_core::io::{FileHandle, IoResult};
use profluens_core::log;
use profluens_core::log::Level;
use profluens_core::time::Timestamp;

/// Cap on the response header block, so a server that never sends `\r\n\r\n` can't
/// make us buffer unboundedly.
const MAX_HEADER_BYTES: usize = 64 * 1024;

/// Cap on a single chunk-size line (`<hex>[;ext]\r\n`) in a chunked body, so a
/// malformed server that streams an unbounded chunk-size/extension line can't make us
/// buffer unboundedly (RFC 9112 §7.1: recipients MUST anticipate large chunk-size
/// numerals and prevent parsing errors — we bound the line instead).
const MAX_CHUNK_LINE_BYTES: usize = 16 * 1024;

/// How many `Location` hops `start()` will follow before giving up (RFC 9110 §15.4:
/// "a client SHOULD detect and intervene in cyclical redirections" — it sets no
/// number, so this is the conventional ten). A cycle is caught by the same cap.
const MAX_REDIRECTS: u8 = 10;

/// Cap on a `Location` field value, so a malicious redirector cannot make us build an
/// unbounded URL string across hops (RFC 9110 §4.1 sets no length limit on a URI, and
/// §2.3 tells recipients to expect longer ones than they generate).
const MAX_URL_BYTES: usize = 8 * 1024;

/// Default dead-air budget on a socket: no bytes for this long is treated as a dead
/// connection and drives a reconnect. Roughly what a podcast client can tolerate
/// before the listener notices; `rtsp/src/client.rs` uses 10 s for its control
/// connection, which is a chattier protocol than a bulk download.
const STALL_TIMEOUT: Duration = Duration::from_secs(20);

/// First reconnect delay; each further consecutive attempt doubles it
/// (500 ms, 1 s, 2 s, 4 s, 8 s), capped at [`RETRY_BACKOFF_CAP`].
const RETRY_BACKOFF_BASE: Duration = Duration::from_millis(500);

/// Ceiling on the exponential backoff.
const RETRY_BACKOFF_CAP: Duration = Duration::from_secs(8);

/// Consecutive reconnect attempts before the download is declared dead. Reset by any
/// byte that arrives, so a flaky link that keeps delivering never exhausts it.
const MAX_RETRIES: u32 = 5;

/// [`Info::total`] sentinel: no total length is known (a chunked or close-delimited
/// response never states one). Not `Option` because the field is an atomic the app
/// can read from another thread while the element runs.
const LEN_UNKNOWN: u64 = u64::MAX;

/// Emits the downloaded body as a raw-byte stream (framing is HTTP's concern, not the
/// pipeline's) — offer the open `bytes` family, no fields.
static OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];

static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &OFFERS,
    dynamic: false,
    validate: None,
}];

static DESC: ElementDesc = ElementDesc {
    name: "httpsrc",
    pads: &PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::None,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

/// What the response head told us about the resource, published for the app.
///
/// A controller that wants to seek needs two facts this element learns only once the
/// first response head is in: how long the resource is, and whether the server serves
/// ranges. `filesrc`'s equivalent — the file length — is something the app can just
/// `stat` for itself before it builds the pipeline (which is exactly what
/// `play/src/player.rs` does to fill [`SeekIndex::file_len`]); over HTTP nobody but
/// this element ever sees it, so it is handed back through a cheap shared handle
/// taken *before* the element is moved into the pipeline:
///
/// ```ignore
/// let src = HttpSrc::new(url);
/// let info = src.info();               // clone the handle, then hand the element over
/// let src = pipeline.add(src);
/// // ... after preroll/start, once the head has been read:
/// if let Some(total) = info.total_bytes() {
///     pipeline.set_seek_index(SeekIndex { entries: Vec::new(), file_len: Some(total) });
/// }
/// ```
///
/// The same shape as [`SeekHandle`]/[`StopHandle`]: an `Arc` of atomics, no new core
/// machinery and nothing to poll on the hot path. Fields read as "not known yet"
/// until `start()` has parsed a response head.
///
/// [`SeekIndex::file_len`]: profluens_core::pipeline::SeekIndex::file_len
/// [`SeekHandle`]: profluens_core::pipeline::SeekHandle
/// [`StopHandle`]: profluens_core::pipeline::StopHandle
#[derive(Clone)]
pub struct HttpInfo {
    inner: Arc<Info>,
}

struct Info {
    /// Total length of the selected representation in bytes, or [`LEN_UNKNOWN`].
    total: AtomicU64,
    /// The server serves byte ranges (`Accept-Ranges: bytes`, or a `206` proved it).
    ranges: AtomicBool,
}

impl HttpInfo {
    // COLD: one handle per element, built in `HttpSrc::new`.
    #[allow(clippy::disallowed_methods)]
    fn new() -> Self {
        Self {
            inner: Arc::new(Info {
                total: AtomicU64::new(LEN_UNKNOWN),
                ranges: AtomicBool::new(false),
            }),
        }
    }

    /// Total size of the resource in bytes — the `Content-Length` of the full
    /// response (RFC 9112 §6.2), or the complete-length of a `206`'s `Content-Range`
    /// (RFC 9110 §14.4). `None` before the first response head is parsed, and for a
    /// chunked or close-delimited body, which never states one.
    ///
    /// This is the byte "duration" a controller needs to turn a proportional seek
    /// (drag to 40%) into the byte offset [`SeekHandle::seek`] wants.
    ///
    /// [`SeekHandle::seek`]: profluens_core::pipeline::SeekHandle::seek
    pub fn total_bytes(&self) -> Option<u64> {
        match self.inner.total.load(Ordering::Acquire) {
            LEN_UNKNOWN => None,
            n => Some(n),
        }
    }

    /// Whether a seek will actually work: the server serves byte ranges *and* the
    /// total length is known. A seek on a stream that is not seekable is answered
    /// with a `Warning` on the bus and otherwise ignored — the download continues
    /// rather than dying (see [`HttpSrc::event`]).
    pub fn is_seekable(&self) -> bool {
        self.inner.ranges.load(Ordering::Acquire) && self.total_bytes().is_some()
    }
}

impl Default for HttpInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// The pieces of an `http://`/`https://` URL this element needs.
struct Url {
    host: String,
    port: u16,
    path: String,
    /// `https://` — wrap the connection in TLS (see [`crate::tls`]).
    tls: bool,
}

impl Url {
    /// `http` or `https` — the scheme a relative reference inherits (RFC 3986 §5.2.2).
    fn scheme(&self) -> &'static str {
        if self.tls {
            "https"
        } else {
            "http"
        }
    }

    /// The default port for this scheme (RFC 9110 §4.2.1, §4.2.2).
    fn default_port(&self) -> u16 {
        if self.tls {
            443
        } else {
            80
        }
    }

    /// The authority as it goes on the wire in `Host` (RFC 9112 §3.2) and in a
    /// recomposed absolute URI (RFC 3986 §5.3): `host`, with `:port` only when the
    /// port is not the scheme's default — virtual hosts on nonstandard ports route
    /// on it, and a redundant `:80` would make two spellings of one origin.
    // COLD: once per connection / per redirect hop.
    #[allow(clippy::disallowed_methods)]
    fn authority(&self) -> String {
        if self.port == self.default_port() {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }

    /// The path without the query — the base for merging a relative reference
    /// (RFC 3986 §5.2.3 merges paths only; §5.2.2 drops the base query).
    fn path_only(&self) -> &str {
        self.path.split('?').next().unwrap_or("/")
    }
}

/// Parse an `http`/`https` URL by hand (the HTTP layer stays dependency-free).
/// Extracts host, port (default 80 for `http`, 443 for `https`), and path (default
/// `/`). Any other scheme is rejected.
// COLD: runs once per connection in `start()` (not the per-chunk body path); the owned
// host/path strings become the `Url` returned to the caller.
#[allow(clippy::disallowed_methods)]
fn parse_http_url(url: &str) -> Result<Url, Error> {
    // The host and path extracted below are formatted straight into the request head
    // in `start()`, so an octet that can end a line there is an injection: a CR or LF
    // in the request target closes the request-line early and everything after it
    // becomes an attacker-chosen field line — or a second request (RFC 9112 §11.1
    // response splitting, §11.2 request smuggling). A bare SP splits the request-line
    // into the wrong three tokens the same way.
    //
    // None of these are legal in a URI to begin with: RFC 3986 §2 builds every
    // component from the unreserved, reserved and percent-encoded sets, so controls,
    // SP and DEL must already have been percent-encoded by whoever produced the URL.
    // Reject rather than escape — a URL is attacker-influenced data in a media
    // framework (playlists, manifests, config), and silently "repairing" one is how a
    // fetch ends up somewhere other than where the caller asked. Bytes >= 0x80 are
    // left alone: also not strictly legal unencoded, but harmless here and common in
    // hand-written UTF-8 paths.
    if let Some(bad) = url.bytes().find(|&b| b < 0x21 || b == 0x7f) {
        return Err(Error::Resource(format!(
            "httpsrc: illegal octet {bad:#04x} in URL {url:?} (RFC 3986 §2: controls, \
             space and DEL must be percent-encoded)"
        )));
    }

    let (rest, tls) = if let Some(rest) = url.strip_prefix("http://") {
        (rest, false)
    } else if let Some(rest) = url.strip_prefix("https://") {
        (rest, true)
    } else {
        return Err(Error::Resource(format!(
            "httpsrc: only http:// and https:// URLs supported: {url}"
        )));
    };

    // Split authority from the path (everything from the first '/'); also stop the
    // authority at '?' or '#' for a bare `http://host?q` form.
    let (authority, path) = match rest.find(['/', '?', '#']) {
        Some(i) if rest.as_bytes()[i] == b'/' => (&rest[..i], rest[i..].to_string()),
        Some(i) => (&rest[..i], format!("/{}", &rest[i..])),
        None => (rest, "/".to_string()),
    };

    if authority.is_empty() {
        return Err(Error::Resource(format!("httpsrc: missing host in {url}")));
    }

    // Strip userinfo if present (`user:pass@host`), then split host:port.
    let hostport = authority.rsplit_once('@').map_or(authority, |(_, hp)| hp);
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => {
            let port = p
                .parse::<u16>()
                .map_err(|_| Error::Resource(format!("httpsrc: invalid port in {url}")))?;
            (h, port)
        }
        None => (hostport, if tls { 443u16 } else { 80u16 }),
    };

    if host.is_empty() {
        return Err(Error::Resource(format!("httpsrc: missing host in {url}")));
    }

    Ok(Url {
        host: host.to_string(),
        port,
        path,
        tls,
    })
}

/// Case-insensitive prefix test (schemes are case-insensitive, RFC 3986 §3.1).
fn starts_with_ci(s: &str, prefix: &str) -> bool {
    s.len() >= prefix.len() && s.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

/// Resolve a `Location` field value against the URI the request was sent to
/// (RFC 3986 §5 "Reference Resolution"), returning an absolute `http`/`https` URL.
///
/// RFC 9110 §10.2.2 defines `Location` as a **URI-reference**, and while it says a
/// sender SHOULD send an absolute URI, it also spells out that a relative one is
/// resolved "relative to the target URI" — and servers send relative ones constantly.
/// The subset implemented is the one that reaches a client in practice, all four from
/// §4.2's grammar:
///
/// * **absolute-URI** — `https://cdn.example/ep1.mp3`, taken whole;
/// * **network-path** (`//host/path`) — inherits the base scheme (§5.2.2's
///   "if defined(R.authority)" branch), which is how a site moves a redirect between
///   `http` and `https` without naming either;
/// * **absolute-path** (`/path`) — base scheme and authority, reference path;
/// * **relative-path** (`ep1.mp3`, `../ep1.mp3`) — merged with the base path per
///   §5.2.3 and then run through [`remove_dot_segments`] (§5.2.4).
///
/// A reference naming any other scheme is refused rather than guessed at: this element
/// speaks HTTP, and a `Location: ftp://…` (or `javascript:…`) is not something to
/// follow. Hostile input is the norm here — every branch below is total, and the
/// result is re-parsed by [`parse_http_url`], which is what rejects the control
/// characters, the missing host and the bad port.
// COLD: at most `MAX_REDIRECTS` times per connection, never on the body path.
#[allow(clippy::disallowed_methods)]
fn resolve_reference(base: &str, reference: &str) -> Result<String, Error> {
    // §3.5: the fragment identifies a secondary resource *client-side* and is never
    // sent to a server, so it is stripped before the reference is resolved (§5.3
    // recomposes it separately; we have nothing to recompose it into).
    let reference = std::str::from_utf8(trim_ascii(reference.as_bytes())).unwrap_or("");
    let reference = reference.split('#').next().unwrap_or("");
    if reference.is_empty() {
        return Err(Error::Resource(format!(
            "httpsrc: {base} redirected with an empty Location (RFC 9110 §10.2.2 \
             requires a URI-reference)"
        )));
    }
    if reference.len() > MAX_URL_BYTES {
        return Err(Error::Resource(format!(
            "httpsrc: {base} redirected to a Location of {} bytes (cap {MAX_URL_BYTES})",
            reference.len()
        )));
    }

    // §4.3 absolute-URI, in the two schemes we speak.
    if starts_with_ci(reference, "http://") || starts_with_ci(reference, "https://") {
        return normalise_absolute(reference);
    }

    let b = parse_http_url(base)?;

    // §4.2 network-path reference: authority from the reference, scheme from the base.
    if let Some(rest) = reference.strip_prefix("//") {
        return normalise_absolute(&format!("{}://{rest}", b.scheme()));
    }

    // Any other `scheme:` prefix (§3.1: scheme, then ':', before any '/' or '?').
    if let Some(colon) = reference.find(':') {
        if !reference[..colon].contains(['/', '?']) {
            return Err(Error::Resource(format!(
                "httpsrc: {base} redirected to {reference:?} — only http:// and https:// \
                 redirects are followed"
            )));
        }
    }

    let authority = b.authority();
    let scheme = b.scheme();

    // §4.2 absolute-path reference.
    if reference.starts_with('/') {
        return normalise_absolute(&format!("{scheme}://{authority}{reference}"));
    }

    // §5.2.2 works on the reference's *components*: an empty reference path keeps the
    // base path and takes only the reference's query — which is what makes `?y` mean
    // "this same document, different query" rather than "the directory, query y".
    let (ref_path, ref_query) = match reference.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (reference, None),
    };
    let base_path = b.path_only();
    let merged = if ref_path.is_empty() {
        base_path.to_string()
    } else {
        // §5.2.3 merge: everything up to and including the base path's last '/', then
        // the reference. With an authority and an empty base path the merge result is
        // the reference with a '/' prefixed.
        let cut = base_path.rfind('/').map_or(0, |i| i + 1);
        if cut == 0 {
            format!("/{ref_path}")
        } else {
            format!("{}{ref_path}", &base_path[..cut])
        }
    };
    match ref_query {
        Some(q) => normalise_absolute(&format!("{scheme}://{authority}{merged}?{q}")),
        None => normalise_absolute(&format!("{scheme}://{authority}{merged}")),
    }
}

/// Validate an absolute `http`/`https` URL and recompose it (RFC 3986 §5.3) with its
/// path run through [`remove_dot_segments`] — §5.2.2 applies that to every resolved
/// reference, including one that arrived absolute, so `http://h/a/../b` and
/// `http://h/b` do not become two different effective URLs to reconnect to.
// COLD: per redirect hop.
#[allow(clippy::disallowed_methods)]
fn normalise_absolute(url: &str) -> Result<String, Error> {
    let u = parse_http_url(url)?;
    let (path, query) = match u.path.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (u.path.as_str(), None),
    };
    let path = remove_dot_segments(path);
    Ok(match query {
        Some(q) => format!("{}://{}{path}?{q}", u.scheme(), u.authority()),
        None => format!("{}://{}{path}", u.scheme(), u.authority()),
    })
}

/// RFC 3986 §5.2.4 `remove_dot_segments`: interpret and remove the `.` and `..`
/// complete path segments, so the result is the path the origin server is actually
/// being asked for. Transcribed from the RFC's own five-case loop (A–E).
///
/// Total by construction — the loop only ever moves bytes from `input` to `out`, and
/// case C's "remove the last segment from output" truncates at a '/' or to empty, so
/// a reference of `../../../../..` walks the output to `/` and stops rather than
/// escaping the root (§5.2.4 notes this is deliberate: "the dot segments are removed
/// … in order to prevent them from being mistaken for a relative reference").
// COLD: per redirect hop; the one output string is the resolved path.
#[allow(clippy::disallowed_methods)]
fn remove_dot_segments(path: &str) -> String {
    let mut input = path;
    let mut out = String::with_capacity(path.len());
    while !input.is_empty() {
        // A. leading "../" or "./" on a relative path: drop the prefix.
        if let Some(rest) = input.strip_prefix("../") {
            input = rest;
        } else if let Some(rest) = input.strip_prefix("./") {
            input = rest;
        // B. "/./" → "/…": dropping two octets leaves the '/' the RFC re-prepends.
        } else if input.starts_with("/./") {
            input = &input[2..];
        } else if input == "/." {
            input = "/";
        // C. "/../" → "/…", and the last segment already moved to the output goes.
        } else if input.starts_with("/../") {
            pop_segment(&mut out);
            input = &input[3..];
        } else if input == "/.." {
            pop_segment(&mut out);
            input = "/";
        // D. a bare "." or ".." with nothing else: drop it.
        } else if input == "." || input == ".." {
            input = "";
        // E. move the first path segment — its leading '/' plus everything up to (not
        //    including) the next '/' — from input to output.
        } else {
            let start = usize::from(input.starts_with('/'));
            let seg_end = input[start..].find('/').map_or(input.len(), |i| start + i);
            out.push_str(&input[..seg_end]);
            input = &input[seg_end..];
        }
    }
    out
}

/// RFC 3986 §5.2.4 case C's "remove the last segment and its preceding '/' (if any)
/// from the output buffer".
fn pop_segment(out: &mut String) {
    match out.rfind('/') {
        Some(i) => out.truncate(i),
        None => out.clear(),
    }
}

/// State of the chunked-body decoder (RFC 9112 §7.1). A chunk is
/// `chunk-size [ chunk-ext ] CRLF chunk-data CRLF`; the body ends with a `0`-size
/// last-chunk, an optional trailer section, and a final CRLF. This element decodes
/// the framing and ignores trailer fields (consuming through the terminating CRLF).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ChunkState {
    /// At the start of a chunk: expecting a `chunk-size [ chunk-ext ] CRLF` line.
    Size,
    /// Inside `chunk-data`: `n` octets of payload remain before its trailing CRLF.
    Data(u64),
    /// Consumed a chunk's `chunk-data`; expecting the CRLF that terminates it.
    DataCrlf,
    /// Saw the zero last-chunk; consuming `trailer-section CRLF` (any trailer field
    /// lines, then the final empty line) up to and including the terminating CRLF.
    Trailer,
    /// Body fully decoded — the terminating CRLF has been consumed.
    Done,
}

/// How the response body is framed (RFC 9112 §6.3), decided in `start()` from the
/// response headers and driven by `process()`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Framing {
    /// `Content-Length: N` (§6.2): this many body octets remain to be emitted.
    Length(u64),
    /// `Transfer-Encoding: chunked` (§6.1, §7.1): decode chunk framing.
    Chunked(ChunkState),
    /// Neither header present (§6.3 rule 8): read to EOF (`Connection: close`).
    Eof,
}

/// ```text
/// +--------------------+
/// |               _____|
/// |  HttpSrc     | src |----> bytes
/// |               ^^^^^|
/// +--------------------+
/// ```
pub struct HttpSrc {
    /// The URL as the caller gave it — what error messages name, and the base the
    /// first `Location` is resolved against.
    url: String,
    /// Where the resource actually lives: `url`, or the last `Location` after
    /// following redirects (RFC 9110 §15.4). Every reconnect and every ranged request
    /// goes here, so a resume does not walk the redirect chain again (and does not
    /// risk a redirector that load-balances to a *different* origin mid-download).
    effective_url: String,
    /// The socket fd, registered with the reactor once `start()` succeeds. `None`
    /// until then (and after `stop`).
    file: Option<FileHandle>,
    /// Undecoded body bytes already pulled from the socket but not yet consumed by the
    /// framing decoder. Seeded in `start()` with the bytes that arrived in the same
    /// read as the header terminator (a Content-Length/chunked body often begins in
    /// that read), then refilled by completed `Recv` ops as the decoder needs more.
    inbuf: Vec<u8>,
    /// Read cursor into `inbuf`: bytes `[..cursor]` are consumed. We advance the
    /// cursor instead of draining so the decoder can work in slices without shifting
    /// the buffer on every step; it is compacted when refilling.
    cursor: usize,
    /// How the body is framed, chosen from the response headers in `start()`.
    framing: Framing,
    /// Set once the body is fully delivered, so `process()` reports `Flow::Eos`.
    done: bool,
    /// How many `Recv` ops are outstanding. At most one is ever *submitted* (a socket
    /// stream is sequential; see the module docs) — but a reconnect leaves the read it
    /// abandoned still in flight under an async backend, and that op's completion must
    /// be accounted for before a new read may go out, or two reads would race on the
    /// fresh socket and permute the byte stream. Hence a count, not a bool.
    in_flight: u32,
    /// The first socket error seen while draining completions, deferred to `process()`
    /// so the drain loop still recycles every buffer in the mailbox before we act on
    /// it. `WouldBlock`/`TimedOut` here is the [`SO_RCVTIMEO`](STALL_TIMEOUT) stall.
    io_err: Option<std::io::ErrorKind>,
    /// Set once a completed `Recv` returned 0 bytes: the peer closed the socket
    /// (EOF). The decoder treats an exhausted `inbuf` under this flag as end-of-body.
    socket_eof: bool,
    /// Monotonic tag on each submitted `Recv` (rides back as the completion `user`).
    /// Reads submitted below `valid_from` are stale (a flush landed) and discarded —
    /// the same seq-floor `filesrc` uses, needed because `discard_buffers` does not
    /// clear the IO mailbox (an in-flight completion still arrives).
    seq: u64,
    /// Completions carrying a `user` below this are dropped: they were submitted
    /// against a connection (or a stream position) that no longer exists — a flush, a
    /// seek, or a reconnect. Bumped to `seq` at each of those.
    valid_from: u64,
    /// Byte offset **in the representation** of the next body octet to be pushed:
    /// every octet below it has already gone downstream. This is the whole of the
    /// offset accounting — the resume point of a dropped connection, the value a
    /// `Range` asks from, and what a seek overwrites. It counts *decoded* body bytes,
    /// so chunk framing and TLS records never enter into it.
    pos: u64,
    /// Total length of the representation, once a response head has stated one.
    total_len: Option<u64>,
    /// The origin serves byte ranges: `Accept-Ranges: bytes` (RFC 9110 §14.3), or a
    /// `206` that proved it. Without this a dropped connection cannot be resumed and
    /// a seek cannot be honoured.
    ranges_ok: bool,
    /// A strong `ETag` from the first response, replayed as `If-Range` on a resume so
    /// a representation that changed underneath us fails loudly (RFC 9110 §14.5).
    etag: Option<String>,
    /// A (re)connect owed before any more body can be decoded: a seek target, or the
    /// resume point of a connection that died. Executed at the top of `process()`.
    pending: Option<Pending>,
    /// Consecutive failed reconnects. Reset by any byte that arrives.
    retries: u32,
    /// Dead-air budget, applied as `SO_RCVTIMEO`/`SO_SNDTIMEO` and as the connect
    /// timeout. [`STALL_TIMEOUT`] unless [`HttpSrc::stall_timeout`] changed it.
    stall: Duration,
    /// The next buffer pushed starts a *new* place in the stream (post-seek), so it
    /// carries [`BufferFlags::DISCONT`]. A resume is contiguous and sets nothing.
    discont: bool,
    /// Which framing element a truncated chunked body was cut in the middle of, for
    /// the message on the way out (`""` until something is cut).
    cut: &'static str,
    /// Facts about the resource for the app (see [`HttpInfo`]).
    info: HttpInfo,
    /// `https://` only: the TLS state machine, handshaken in `start()` (see
    /// [`crate::tls`]). After that it never touches the socket — completed `Recv`
    /// ciphertext is fed through it and the plaintext lands in `inbuf`. `None` for
    /// plain `http://`.
    tls: Option<rustls::ClientConnection>,
    /// The peer sent `close_notify` (RFC 8446 §6.1): the TLS stream ended
    /// *authenticated*. Without this, a socket EOF under TLS is just a TCP FIN
    /// anyone on the path could have forged — fatal for an EOF-delimited body.
    tls_closed: bool,
}

/// A (re)connect `process()` owes before it can decode more body.
#[derive(Clone, Copy, Debug)]
struct Pending {
    /// The representation byte offset the new connection must start at.
    at: u64,
    /// Earliest instant to attempt it — the backoff after a failure. `None` for a
    /// seek, which the user is waiting on and which no failure has yet earned a wait.
    not_before: Option<Instant>,
}

impl HttpSrc {
    // COLD: one-time constructor; `inbuf` is the reused body-buffer grown/compacted per read.
    #[allow(clippy::disallowed_methods)]
    pub fn new(url: impl Into<String>) -> Self {
        let url = url.into();
        Self {
            effective_url: url.clone(),
            url,
            file: None,
            inbuf: Vec::new(),
            cursor: 0,
            // Overwritten by `start()`; `Eof` is the conservative default.
            framing: Framing::Eof,
            done: false,
            in_flight: 0,
            io_err: None,
            socket_eof: false,
            seq: 0,
            valid_from: 0,
            pos: 0,
            total_len: None,
            ranges_ok: false,
            etag: None,
            pending: None,
            retries: 0,
            stall: STALL_TIMEOUT,
            discont: false,
            cut: "",
            info: HttpInfo::new(),
            tls: None,
            tls_closed: false,
        }
    }

    /// Override the dead-air budget (20 s by default): the socket receive and send
    /// timeouts, and the connect timeout. A shorter one makes a stalled server
    /// reconnect sooner at the price of giving up on a genuinely slow one.
    ///
    /// The receive half is `SO_RCVTIMEO`, which only bites a *blocking* read — the
    /// synchronous reactor's `read(2)`. An io_uring `Recv` is not covered by it; that
    /// needs a reactor timeout op (`IORING_OP_LINK_TIMEOUT`), which the [`Reactor`]
    /// trait does not expose yet.
    ///
    /// [`Reactor`]: profluens_core::io::Reactor
    #[must_use]
    pub fn stall_timeout(mut self, timeout: Duration) -> Self {
        self.stall = timeout;
        self
    }

    /// A handle on what the response head says about the resource — total length and
    /// range support (see [`HttpInfo`]). Take it *before* handing the element to the
    /// pipeline; it stays valid for the element's whole life.
    pub fn info(&self) -> HttpInfo {
        self.info.clone()
    }

    /// Undecoded body bytes not yet consumed by the decoder.
    fn avail(&self) -> &[u8] {
        &self.inbuf[self.cursor..]
    }

    /// Submit one streaming socket read if the fd is registered, none is already in
    /// flight, and the peer has not closed. First compacts `inbuf` by dropping the
    /// consumed prefix `[..cursor]` (so it can't grow unbounded across chunks). The
    /// completed bytes are appended to `inbuf` in `process()`; the pooled buffer the
    /// reactor filled is recycled. Returns whether a read was submitted.
    fn submit_read(&mut self, ctx: &mut Ctx) -> bool {
        // Nothing may go out while a read is outstanding (sequential stream), while
        // the peer is done, or while a reconnect is owed — in the last case the fd
        // the reactor holds is about to be replaced, and a read submitted now would
        // be executed against whichever socket wins the race.
        if self.in_flight > 0 || self.socket_eof || self.pending.is_some() {
            return false;
        }
        let Some(file) = self.file else {
            return false;
        };
        if self.cursor > 0 {
            self.inbuf.drain(..self.cursor);
            self.cursor = 0;
        }
        // A pooled slot to read into; `None` is pool backpressure — retry next pass.
        let buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return false,
        };
        let user = self.seq;
        self.seq += 1;
        ctx.io().submit_recv(file, buf, user);
        self.in_flight += 1;
        true
    }

    /// Drain any completed `Recv`, appending its bytes to `inbuf` (or noting EOF).
    /// Returns `true` if fresh input landed, so the caller can try the decoder again.
    /// Stale completions (below the seq floor after a flush, a seek or a reconnect)
    /// are discarded — but still counted down, since the in-flight budget is what
    /// keeps a second read off the socket.
    ///
    /// A socket error is recorded rather than raised: the rest of the mailbox is
    /// drained first (every completion carries a pooled buffer that has to be
    /// recycled), and `process()` decides between reconnecting and failing.
    fn drain_reads(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        let mut got = false;
        loop {
            let Some(c) = ctx.io().next_completion() else {
                break;
            };
            self.in_flight = self.in_flight.saturating_sub(1);
            if c.user < self.valid_from {
                continue; // submitted before a flush/seek/reconnect: discard (buf recycles)
            }
            match c.result {
                IoResult::Ok(0) => self.socket_eof = true, // peer closed; buf recycles
                IoResult::Ok(_) => {
                    // Bytes arrived: the connection is alive, so the consecutive-failure
                    // count starts over (a link that drops every few megabytes must not
                    // exhaust the retry budget over the course of an hour-long episode).
                    self.retries = 0;
                    // The reactor `set_len`'d the buffer to the bytes read, so `data()`
                    // is exactly those bytes. Over TLS those bytes are ciphertext and
                    // go through the decrypt stage; plain HTTP appends them directly.
                    if self.tls.is_some() {
                        got |= self.decrypt_into_inbuf(c.buf.memory.data())?;
                    } else {
                        self.inbuf.extend_from_slice(c.buf.memory.data());
                        got = true;
                    }
                }
                IoResult::Cancelled => {}
                IoResult::Err(k) => {
                    self.io_err.get_or_insert(k);
                }
            }
        }
        Ok(got)
    }

    /// Feed reactor-completed **ciphertext** through the TLS state machine, appending
    /// decrypted plaintext to `inbuf`. Returns whether any plaintext landed. Sans-IO:
    /// `read_tls` consumes from the in-memory slice, `process_new_packets` advances
    /// the session, and the `reader()` hands out exactly the plaintext it reports —
    /// the socket itself is never touched here. A `close_notify` alert marks the TLS
    /// stream authenticated-ended (`tls_closed`), which the EOF-framed decoder
    /// requires before trusting a socket EOF; bytes after it are ignored (RFC 8446
    /// §6.1: data after close_notify must be disregarded).
    // `Read::read_exact` here drains rustls's in-memory plaintext buffer — no fd, no
    // syscall, cannot block — not the element streaming IO the workspace lint bans.
    #[allow(clippy::disallowed_methods)]
    fn decrypt_into_inbuf(&mut self, mut src: &[u8]) -> Result<bool, Error> {
        let Self {
            tls,
            inbuf,
            tls_closed,
            socket_eof,
            url,
            ..
        } = self;
        let conn = tls.as_mut().expect("decrypt_into_inbuf without a TLS session");
        let mut got = false;
        while !src.is_empty() && !*tls_closed {
            // Reading from a slice cannot fail; `Ok(0)` would mean the deframer
            // buffer is full, which cannot persist since every pass below drains all
            // plaintext — bail rather than spin if it ever happens.
            let n = conn
                .read_tls(&mut src)
                .map_err(|e| Error::Resource(format!("httpsrc: TLS read from {url}: {e}")))?;
            if n == 0 {
                return Err(Error::Resource(format!(
                    "httpsrc: TLS deframer stalled on {url}"
                )));
            }
            let state = conn
                .process_new_packets()
                .map_err(|e| Error::Resource(format!("httpsrc: TLS error from {url}: {e}")))?;
            let plain = state.plaintext_bytes_to_read();
            if plain > 0 {
                let start = inbuf.len();
                inbuf.resize(start + plain, 0);
                conn.reader()
                    .read_exact(&mut inbuf[start..])
                    .map_err(|e| Error::Resource(format!("httpsrc: TLS read from {url}: {e}")))?;
                got = true;
            }
            if state.peer_has_closed() {
                *tls_closed = true;
                *socket_eof = true; // the TLS stream is over, whatever TCP does next
            }
        }
        Ok(got)
    }

    /// `Content-Length` body (RFC 9112 §6.2): emit up to `dst.len()` of the remaining
    /// octets from buffered input, then signal EOS once the declared count is
    /// exhausted — without waiting for the socket to close. Consumes only what is
    /// buffered; when `inbuf` runs dry before the length is met, returns
    /// [`NeedMore`](FillOutcome::NeedMore) so `process()` submits another read.
    fn fill_length(&mut self, dst: &mut [u8]) -> Result<FillOutcome, Error> {
        let Framing::Length(remaining) = self.framing else {
            unreachable!("fill_length called for non-length framing");
        };
        if remaining == 0 {
            return Ok(FillOutcome::Done(0, true));
        }
        // How many octets we may take this step: bounded by both the output buffer and
        // the declared remaining length. `usize::try_from` guards a >usize length on
        // 32-bit targets (we just cap at the buffer size, which is always fine).
        let want = dst
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));

        let buffered = self.avail().len();
        if buffered == 0 {
            if self.socket_eof {
                // The connection ended before the declared length arrived (RFC 9112
                // §6.3 rule 6: the message is incomplete) — the classic mid-download
                // drop. Resumable when the origin serves ranges, since `Content-Length`
                // makes it unambiguous how much is still owed.
                return Ok(FillOutcome::Broken);
            }
            return Ok(FillOutcome::NeedMore);
        }
        let take = buffered.min(want);
        dst[..take].copy_from_slice(&self.avail()[..take]);
        self.cursor += take;

        let left = remaining - take as u64;
        self.framing = Framing::Length(left);
        Ok(FillOutcome::Done(take, left == 0))
    }

    /// Close-delimited body (RFC 9112 §6.3 rule 8): emit buffered leftover, ending the
    /// download when the socket reports EOF and no buffered bytes remain. Over TLS the
    /// EOF must be the peer's `close_notify` (RFC 8446 §6.1) — a bare TCP FIN is not
    /// authenticated, and for a body whose *only* length signal is EOF, accepting it
    /// would let an on-path attacker silently truncate the download.
    fn fill_eof(&mut self, dst: &mut [u8]) -> Result<FillOutcome, Error> {
        let buffered = self.avail().len();
        if buffered > 0 {
            let take = buffered.min(dst.len());
            dst[..take].copy_from_slice(&self.avail()[..take]);
            self.cursor += take;
            return Ok(FillOutcome::Done(take, false));
        }
        if self.socket_eof {
            if self.tls.is_some() && !self.tls_closed {
                return Err(Error::Resource(format!(
                    "httpsrc: {} TLS stream truncated (connection closed without \
                     close_notify on a close-delimited body)",
                    self.url
                )));
            }
            // A *clean* EOF is the framing (RFC 9112 §6.3 rule 8) — the body is
            // complete by definition. An error or a stall is not: nothing distinguishes
            // a cut close-delimited body from a whole one, so it goes to the resume
            // path, which will only get anywhere if the origin serves ranges.
            if self.io_err.is_some() {
                return Ok(FillOutcome::Broken);
            }
            return Ok(FillOutcome::Done(0, true));
        }
        Ok(FillOutcome::NeedMore)
    }

    /// `chunked` transfer coding (RFC 9112 §7.1). Runs the chunk state machine over
    /// buffered input and writes as many decoded `chunk-data` octets into `dst` as fit.
    /// Returns [`FillOutcome`]: `Done(written, eos)` when it makes progress or reaches
    /// the end, or [`NeedMore`](FillOutcome::NeedMore) when the decoder needs bytes
    /// that are not buffered yet (so `process()` submits another read). `eos` is set
    /// once the zero last-chunk and its terminating CRLF have been consumed.
    fn fill_chunked(&mut self, dst: &mut [u8]) -> Result<FillOutcome, Error> {
        let mut written = 0;
        loop {
            let state = match self.framing {
                Framing::Chunked(s) => s,
                _ => unreachable!("fill_chunked called for non-chunked framing"),
            };

            match state {
                ChunkState::Done => return Ok(FillOutcome::Done(written, true)),

                ChunkState::Size => {
                    // Need a full `chunk-size [ chunk-ext ] CRLF` line before decoding.
                    match self.take_line(MAX_CHUNK_LINE_BYTES)? {
                        LineResult::Line(len) => {
                            // Parse the line in place (no copy), then consume it + CRLF.
                            let size = parse_chunk_size(&self.avail()[..len], &self.url)?;
                            self.cursor += len + 2;
                            self.framing = Framing::Chunked(if size == 0 {
                                ChunkState::Trailer // zero = last-chunk (§7.1)
                            } else {
                                ChunkState::Data(size)
                            });
                        }
                        LineResult::NeedMore => {
                            return self.need_more_or_short(written, "chunk-size line");
                        }
                    }
                }

                ChunkState::Data(remaining) => {
                    // Copy as much chunk-data as fits in `dst` and is buffered.
                    let space = dst.len() - written;
                    if space == 0 {
                        return Ok(FillOutcome::Done(written, false)); // full; resume next step
                    }
                    let buffered = self.avail().len();
                    if buffered == 0 {
                        return self.need_more_or_short(written, "chunk-data");
                    }
                    let take = space
                        .min(buffered)
                        .min(usize::try_from(remaining).unwrap_or(usize::MAX));
                    dst[written..written + take].copy_from_slice(&self.avail()[..take]);
                    self.cursor += take;
                    written += take;
                    let left = remaining - take as u64;
                    self.framing = Framing::Chunked(if left == 0 {
                        ChunkState::DataCrlf // chunk-data consumed; expect its CRLF
                    } else {
                        ChunkState::Data(left)
                    });
                }

                ChunkState::DataCrlf => {
                    // Consume the CRLF that terminates a chunk's data (§7.1 grammar:
                    // `chunk = chunk-size [ chunk-ext ] CRLF chunk-data CRLF`).
                    match self.take_crlf()? {
                        Some(true) => self.framing = Framing::Chunked(ChunkState::Size),
                        Some(false) => {
                            return Err(Error::Resource(format!(
                                "httpsrc: {} missing CRLF after chunk-data",
                                self.url
                            )));
                        }
                        None => return self.need_more_or_short(written, "chunk CRLF"),
                    }
                }

                ChunkState::Trailer => {
                    // After the zero last-chunk: consume `trailer-section CRLF`, i.e.
                    // zero or more `field-line CRLF` then the terminating empty line
                    // (§7.1.2). We ignore trailer field contents entirely.
                    match self.take_line(MAX_CHUNK_LINE_BYTES)? {
                        LineResult::Line(len) => {
                            self.cursor += len + 2; // consume the line and its CRLF
                            if len == 0 {
                                // The empty line terminates the body.
                                self.framing = Framing::Chunked(ChunkState::Done);
                                return Ok(FillOutcome::Done(written, true));
                            }
                            // A trailer field line — skip it and keep reading.
                        }
                        LineResult::NeedMore => {
                            return self.need_more_or_short(written, "chunked trailer");
                        }
                    }
                }
            }

            // If we filled the output buffer, hand it off; otherwise loop to make more
            // progress (the "need more input" returns above are what bound the loop).
            if written == dst.len() {
                return Ok(FillOutcome::Done(written, false));
            }
        }
    }

    /// The decoder needs more bytes than are buffered. If the socket already ended,
    /// the body was truncated mid-framing — hand out whatever was decoded first, then
    /// report [`Broken`](FillOutcome::Broken) on the next pass so the resume path can
    /// try (`what` names the framing element that was cut, for the message). Otherwise
    /// return whatever was decoded so far and ask for another read — emitting the
    /// partial output keeps buffers flowing while the next read is in flight.
    fn need_more_or_short(
        &mut self,
        written: usize,
        what: &'static str,
    ) -> Result<FillOutcome, Error> {
        if written > 0 {
            return Ok(FillOutcome::Done(written, false));
        }
        if self.socket_eof {
            self.cut = what;
            return Ok(FillOutcome::Broken);
        }
        Ok(FillOutcome::NeedMore)
    }

    /// Locate one CRLF-terminated line in buffered input, *without* the CRLF, returning
    /// its length (the caller reads `self.avail()[..len]` in place and advances the
    /// cursor by `len + 2` once done — no heap copy on the hot chunked path). Returns
    /// [`LineResult::NeedMore`] if no complete line is buffered yet. Bounds the scanned
    /// length by `max` (RFC 9112 §7.1: guard against unbounded chunk-size / extension
    /// lines) — a longer line is a protocol error.
    fn take_line(&self, max: usize) -> Result<LineResult, Error> {
        let buf = self.avail();
        if let Some(i) = find_crlf(buf) {
            Ok(LineResult::Line(i))
        } else {
            if buf.len() > max {
                return Err(Error::Resource(format!(
                    "httpsrc: chunk line exceeded {max} bytes from {} (malformed)",
                    self.url
                )));
            }
            Ok(LineResult::NeedMore)
        }
    }

    /// Try to consume a leading CRLF from buffered input. `Some(true)` = consumed;
    /// `Some(false)` = the next two bytes are present but are not CRLF (protocol
    /// error); `None` = fewer than two bytes buffered, need to read more.
    fn take_crlf(&mut self) -> Result<Option<bool>, Error> {
        let buf = self.avail();
        if buf.len() < 2 {
            return Ok(None);
        }
        if &buf[..2] == b"\r\n" {
            self.cursor += 2;
            Ok(Some(true))
        } else {
            Ok(Some(false))
        }
    }

    // -----------------------------------------------------------------------------
    // Connection lifecycle: connect, redirects, ranges, resume.
    // -----------------------------------------------------------------------------

    /// Whether a seek can be honoured: the origin serves byte ranges *and* the total
    /// length is known. Both halves are needed — without ranges there is no way to
    /// start anywhere but zero, and without a length a controller has no byte offset
    /// to ask for in the first place.
    fn seekable(&self) -> bool {
        self.ranges_ok && self.total_len.is_some()
    }

    /// Publish what the head told us, for the app (see [`HttpInfo`]).
    fn publish_info(&self) {
        self.info
            .inner
            .total
            .store(self.total_len.unwrap_or(LEN_UNKNOWN), Ordering::Release);
        self.info.inner.ranges.store(self.ranges_ok, Ordering::Release);
    }

    /// Post a `Warning` on the bus. Warnings are droppable (`core/src/bus.rs`), which
    /// is right for these: a redirect that downgraded the scheme, or a reconnect that
    /// is already being handled — facts an app may want to surface, none of which the
    /// pipeline should die of.
    // COLD: per redirect / per reconnect, never per buffer.
    #[allow(clippy::disallowed_methods)]
    fn warn(&self, ctx: &mut Ctx, message: String) {
        let element = ctx.element();
        ctx.post(BusMessage::Warning {
            element,
            error: Error::Element { element, message },
        });
    }

    /// Build the request head. `range_from` asks for the remainder of the
    /// representation from that byte on.
    // COLD: one request head per connection.
    #[allow(clippy::disallowed_methods)]
    fn build_request(&self, url: &Url, range_from: Option<u64>) -> String {
        // `Accept-Encoding: identity` asks for the resource uncoded. It has to be said
        // out loud: RFC 9110 §12.5.3 rule 1 makes *every* content coding acceptable to a
        // client that sends no `Accept-Encoding`, and we implement no content decoders
        // (see `check_content_coding`). `Host` carries the port when it isn't the
        // scheme default (RFC 9112 §3.2) — virtual hosts on nonstandard ports route on
        // it.
        let mut req = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\
             Accept-Encoding: identity\r\nUser-Agent: profluens\r\n",
            url.path,
            url.authority()
        );
        if let Some(first) = range_from {
            // RFC 9110 §14.2: an open-ended `int-range` — "the remainder of the
            // representation", which is exactly what both a seek and a resume want.
            req.push_str(&format!("Range: bytes={first}-\r\n"));
            // §14.5: `If-Range` makes the range request conditional — the server sends
            // `206` if the validator still matches and the *whole* representation
            // (`200`) if it does not, instead of splicing bytes from a representation
            // that changed since the first half was fetched. Only a strong validator
            // may be sent (see `strong_etag`), and only one field is allowed.
            if let Some(tag) = &self.etag {
                req.push_str(&format!("If-Range: {tag}\r\n"));
            }
        }
        req.push_str("\r\n");
        req
    }

    /// Connect to the effective URL and read a final response head, following
    /// redirects (RFC 9110 §15.4) and updating [`Self::effective_url`] as it goes.
    ///
    /// **This blocks the group thread**, on the first connection and on every
    /// reconnect: DNS resolution, the TCP handshake, the TLS handshake and the header
    /// read are all synchronous, once per hop. That is a deliberate carry-over of the
    /// `start()`-time exception (a source is normally alone at the head of its group,
    /// so the thread it holds up is its own) extended to the reconnect path, where it
    /// is a weaker argument: a group with an inlined decoder stalls with it, and the
    /// backoff sleep is *not* taken here precisely because it would make that worse
    /// (it is a deadline checked between passes instead — see [`Self::run_pending`]).
    /// Making this asynchronous needs the reactor to grow connect/handshake ops; until
    /// then the bound on the damage is [`Self::stall_timeout`], which caps every
    /// blocking step.
    // Blocking connect + handshake + header read: the sanctioned `start()`-time
    // exception to the workspace blocking-IO lint, and the one-time per-connection
    // allocations (request head, header block, effective URL) that go with it.
    #[allow(clippy::disallowed_methods)]
    fn open(&mut self, ctx: &mut Ctx, range_from: Option<u64>) -> Result<Live, Error> {
        let mut hops = 0u8;
        loop {
            let url = parse_http_url(&self.effective_url)?;
            let mut stream = connect_socket(&url, self.stall, &self.effective_url)?;
            // A fresh TLS session per hop and per reconnect: the `Location` may have
            // changed host, and a resumed connection is a new connection.
            let mut tls = if url.tls {
                Some(tls::connect(&url.host, &mut stream, &self.effective_url)?)
            } else {
                None
            };
            let request = self.build_request(&url, range_from);
            let (raw, peer_closed) =
                send_and_read_head(&mut stream, tls.as_mut(), &request, &self.effective_url)?;

            match Disposition::of(raw.status, &self.effective_url, &status_line(&raw.block))? {
                Disposition::Final => {
                    let framing = interpret_head(&raw, &self.effective_url)?;
                    return Ok(Live {
                        stream,
                        tls,
                        tls_closed: peer_closed,
                        status: raw.status,
                        framing,
                        leftover: raw.leftover,
                        accept_ranges: accepts_byte_ranges(&raw.block),
                        content_range: content_range_of(&raw.block),
                        etag: strong_etag(&raw.block),
                    });
                }
                Disposition::Redirect => {
                    hops += 1;
                    if hops > MAX_REDIRECTS {
                        return Err(Error::Resource(format!(
                            "httpsrc: {} exceeded {MAX_REDIRECTS} redirects (RFC 9110 \
                             §15.4: a client should detect and intervene in cyclical \
                             redirections); last hop {}",
                            self.url, self.effective_url
                        )));
                    }
                    // §15.4: the target is in `Location`. Without one there is nothing
                    // to follow — for 301/302/307/308 §10.2.2 makes the field the whole
                    // point of the response.
                    let location = location_of(&raw.block).ok_or_else(|| {
                        Error::Resource(format!(
                            "httpsrc: {} returned HTTP status {} without a Location \
                             (RFC 9110 §10.2.2)",
                            self.effective_url, raw.status
                        ))
                    })?;
                    let next = resolve_reference(&self.effective_url, &location)?;

                    // Both directions are followed — an origin moving to TLS, and a
                    // CDN handing off to a plain-HTTP media host, are both ordinary —
                    // but a *downgrade* silently drops the confidentiality and
                    // authentication the caller asked for by typing `https`, so it is
                    // said out loud rather than buried.
                    let to_tls = starts_with_ci(&next, "https://");
                    if url.tls && !to_tls {
                        let msg = format!(
                            "httpsrc: {} redirected (HTTP {}) to plain HTTP at {next} — \
                             the rest of this download is unencrypted and unauthenticated",
                            self.effective_url, raw.status
                        );
                        self.warn(ctx, msg);
                    }
                    // Fields are POD (`FieldValue::Str` is `&'static str`), so the
                    // resolved URL cannot ride the record — it rides the downgrade
                    // warning above, and the effective URL is in every later message.
                    log!(
                        &*ctx,
                        Level::Debug,
                        "redirect",
                        status = raw.status,
                        hop = hops,
                        tls = to_tls,
                    );
                    self.effective_url = next;
                    // `Connection: close` was requested, so the server is hanging up
                    // anyway; dropping the socket here also discards the redirect's
                    // body unread, which is what we want (it is a courtesy page, and
                    // it may carry a coding we would refuse on a body we had to read).
                    drop(tls);
                    drop(stream);
                }
            }
        }
    }

    /// Adopt a freshly opened connection as the live one, resuming the byte stream at
    /// representation offset `at`. `ranged` says whether the request carried a
    /// `Range`, which is what makes a `200` an error rather than the normal answer.
    // One-time per connection: the leftover body bytes and the fd registration.
    #[allow(clippy::disallowed_methods)]
    fn install(&mut self, ctx: &mut Ctx, live: Live, at: u64, ranged: bool) -> Result<(), Error> {
        let Live {
            stream,
            tls,
            tls_closed,
            status,
            framing,
            leftover,
            accept_ranges,
            content_range,
            etag,
        } = live;

        // What the origin advertises is only a hint (see `accepts_byte_ranges`); the
        // status is the proof, either way.
        self.ranges_ok |= accept_ranges;
        match status {
            206 => {
                // RFC 9110 §14.4: a 206 carries `Content-Range`, and its first-byte-pos
                // is where the part sits in the representation. If that is not the byte
                // we asked to resume at, splicing it onto what we already pushed would
                // silently corrupt the stream — a duplicated or missing span nobody
                // downstream can see. Refuse instead.
                let cr = content_range.ok_or_else(|| {
                    Error::Resource(format!(
                        "httpsrc: {} answered 206 without a usable Content-Range \
                         (RFC 9110 §14.4)",
                        self.effective_url
                    ))
                })?;
                if cr.first != at {
                    return Err(Error::Resource(format!(
                        "httpsrc: {} answered 206 for bytes {}-{} — asked for bytes={at}- \
                         (RFC 9110 §14.4)",
                        self.effective_url, cr.first, cr.last
                    )));
                }
                self.ranges_ok = true;
                if let Some(total) = cr.complete {
                    self.total_len = Some(total);
                }
            }
            _ if ranged => {
                // RFC 9110 §15.3.7: a `206` is the answer to a satisfiable range
                // request; a `200` means the server ignored `Range` entirely (or
                // `If-Range` failed, §14.5) and is sending the whole representation
                // from byte zero. There is no way to make that continue a stream that
                // is already `at` bytes in, so ranges are struck off and this stops —
                // unless byte zero is exactly where we were going anyway.
                self.ranges_ok = false;
                if at != 0 {
                    self.publish_info();
                    return Err(Error::Resource(format!(
                        "httpsrc: {} answered HTTP {status} to `Range: bytes={at}-` — it \
                         does not serve byte ranges (RFC 9110 §14.2), so this download \
                         cannot resume or seek",
                        self.effective_url
                    )));
                }
                self.total_len = length_of(framing);
            }
            _ => {
                // A full response: its `Content-Length` is the length of the whole
                // representation (RFC 9112 §6.2). A chunked or close-delimited body
                // states none, and stays unknown.
                self.total_len = length_of(framing);
            }
        }
        // The first response's validator is the one every later resume is conditional
        // on; a validator picked up from a *later* response would be describing bytes
        // we already have half of.
        if self.etag.is_none() {
            self.etag = etag;
        }
        self.publish_info();

        self.framing = framing;
        self.inbuf = leftover;
        self.cursor = 0;
        self.pos = at;
        self.done = false;
        self.io_err = None;
        self.tls = tls;
        self.tls_closed = tls_closed;
        // A peer that already sent `close_notify` in the handshake read is done; the
        // decoder must see that rather than wait for a read that will never come.
        self.socket_eof = tls_closed;
        // Any completion still owed by the connection this one replaces describes a
        // socket that no longer exists: raise the floor so it is discarded on arrival
        // (`discard_buffers` does not clear the IO mailbox). Its slot is still counted
        // in `in_flight`, so no new read goes out until it has landed — two reads on
        // one socket would permute the byte stream.
        self.valid_from = self.seq;

        // Hand the connected socket to the reactor. `into_raw_fd` consumes the
        // `TcpStream` (releasing sole fd ownership without closing it); `File` adopts
        // that fd, and the reactor owns/closes it thereafter — the same registration
        // path `filesrc` uses, so body reads become submitted ops. Re-registering
        // replaces the element's file and closes the previous one (`core/src/io.rs`
        // `set_file`), which is how the dead socket is disposed of.
        // SAFETY: the fd came from `into_raw_fd`, which transferred ownership to us.
        let file = unsafe { std::fs::File::from_raw_fd(stream.into_raw_fd()) };
        self.file = Some(ctx.io().register(file));

        log!(
            &*ctx,
            Level::Info,
            "open",
            status = status,
            at = at,
            len = self.total_len.unwrap_or(0),
            ranges = self.ranges_ok,
        );
        Ok(())
    }

    /// Execute a pending (re)connect once its backoff has elapsed. Returns whether the
    /// caller may go on to decode this pass.
    fn run_pending(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        let Some(p) = self.pending else {
            return Ok(true);
        };
        if let Some(t) = p.not_before {
            if Instant::now() < t {
                // Still backing off. Returning without progress parks the group, which
                // wakes on its own coarse tick (10 ms) — so the deadline is checked
                // again promptly *and* a stop or a seek is still observed immediately,
                // which a `thread::sleep` of up to 8 s here would have prevented.
                return Ok(false);
            }
        }
        // Byte zero needs no `Range` at all: a plain GET is the same request, and it
        // works on the origins that serve no ranges.
        let ranged = p.at > 0;
        let opened = self
            .open(ctx, ranged.then_some(p.at))
            .and_then(|live| self.install(ctx, live, p.at, ranged));
        match opened {
            Ok(()) => {
                self.pending = None;
                Ok(true)
            }
            Err(e) => {
                self.retry_or_fail(ctx, p.at, e)?;
                Ok(false)
            }
        }
    }

    /// Schedule another attempt at `at`, or give up and hand the failure to the
    /// pipeline (which posts it on the bus and stops the run).
    fn retry_or_fail(&mut self, ctx: &mut Ctx, at: u64, e: Error) -> Result<(), Error> {
        // Nothing to retry *with*: resuming anywhere but byte zero needs ranges, and a
        // server that has already answered a range request with the whole file will
        // answer the next one the same way.
        if at > 0 && !self.ranges_ok {
            return Err(Error::Resource(format!(
                "httpsrc: {} cannot resume at byte {at} — the server does not serve byte \
                 ranges (RFC 9110 §14.2); {} bytes were delivered. Underlying failure: {e:?}",
                self.effective_url, self.pos
            )));
        }
        if self.retries >= MAX_RETRIES {
            return Err(Error::Resource(format!(
                "httpsrc: {} gave up after {MAX_RETRIES} consecutive reconnects at byte \
                 {at}. Last failure: {e:?}",
                self.effective_url
            )));
        }
        self.retries += 1;
        // 500 ms, 1 s, 2 s, 4 s, 8 s — `1 << (n-1)` cannot overflow for n <= MAX_RETRIES,
        // and is saturated anyway so a raised cap stays total.
        let delay = RETRY_BACKOFF_BASE
            .saturating_mul(1u32 << (self.retries - 1).min(16))
            .min(RETRY_BACKOFF_CAP);
        self.pending = Some(Pending {
            at,
            not_before: Some(Instant::now() + delay),
        });
        let msg = format!(
            "httpsrc: {} reconnecting at byte {at} in {} ms (attempt {}/{MAX_RETRIES}): {e:?}",
            self.effective_url,
            delay.as_millis(),
            self.retries
        );
        self.warn(ctx, msg);
        Ok(())
    }

    /// The live connection can deliver no more of this body — a socket error, a stall,
    /// or an EOF with bytes still owed. Resume from [`Self::pos`], or fail.
    // COLD: a dropped connection, not a per-buffer event.
    #[allow(clippy::disallowed_methods)]
    fn on_broken(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        let why = match self.io_err.take() {
            // A blocking `read(2)` on a socket carrying `SO_RCVTIMEO` reports a
            // timeout as `EAGAIN` (`man 7 socket`: "if no data has been transferred
            // ... -1 is returned with errno set to EAGAIN"), which std maps to
            // `WouldBlock`; other platforms/backends prefer `TimedOut`. Neither can
            // mean anything else here — the fd is blocking.
            Some(k @ (std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut)) => {
                format!("stalled: no data for {:?} ({k:?})", self.stall)
            }
            Some(k) => format!("socket error {k:?}"),
            None => match self.framing {
                Framing::Length(owed) => {
                    format!("closed with {owed} of the declared Content-Length unread")
                }
                Framing::Chunked(_) => format!("closed mid {}", self.cut),
                Framing::Eof => "closed".to_string(),
            },
        };
        let at = self.pos;
        let e = Error::Resource(format!(
            "httpsrc: {} {why} after {at} bytes",
            self.effective_url
        ));
        self.retry_or_fail(ctx, at, e)
    }

}

/// A freshly opened connection with its response head parsed, ready to be installed
/// as the live one (see [`HttpSrc::install`]).
struct Live {
    stream: TcpStream,
    /// The handshaken TLS session for an `https://` connection, which stays behind as
    /// the decrypt stage (see the module docs).
    tls: Option<rustls::ClientConnection>,
    /// The peer already sent `close_notify` while the head was read.
    tls_closed: bool,
    status: u16,
    framing: Framing,
    /// Body bytes that arrived in the same read as the header terminator.
    leftover: Vec<u8>,
    /// `Accept-Ranges: bytes` (RFC 9110 §14.3).
    accept_ranges: bool,
    /// `Content-Range` (RFC 9110 §14.4) — present on a `206`.
    content_range: Option<ContentRange>,
    /// A strong `ETag` to make later resumes conditional on (RFC 9110 §14.5).
    etag: Option<String>,
}

/// The representation length a framing states, if it states one.
fn length_of(framing: Framing) -> Option<u64> {
    match framing {
        Framing::Length(n) => Some(n),
        // Chunked and close-delimited bodies declare no length (RFC 9112 §6.3);
        // that is exactly the case where seeking has to be refused.
        Framing::Chunked(_) | Framing::Eof => None,
    }
}

/// The `Location` field value, owned (RFC 9110 §10.2.2).
// COLD: per redirect hop.
#[allow(clippy::disallowed_methods)]
fn location_of(header_block: &[u8]) -> Option<String> {
    let lossy = String::from_utf8_lossy(header_block);
    let text = unfold_obs_fold(&lossy);
    let v = header_value(&text, "Location")?;
    if v.is_empty() {
        return None;
    }
    Some(v.to_string())
}

/// The response's `Content-Range`, if it has a usable one (RFC 9110 §14.4).
fn content_range_of(header_block: &[u8]) -> Option<ContentRange> {
    let lossy = String::from_utf8_lossy(header_block);
    let text = unfold_obs_fold(&lossy);
    parse_content_range(header_value(&text, "Content-Range")?)
}

/// Connect to `url`, with `timeout` bounding every blocking step.
///
/// [`TcpStream::connect`] has no timeout of its own, so the address is resolved first
/// and each candidate connected to with one: a host whose first `AAAA` blackholes must
/// not hold the group thread for the kernel's SYN budget (~130 s). Every address is
/// tried, which is also what makes a dual-stack CDN work from a v4-only network.
///
/// The receive and send timeouts are set here, *before* the fd is handed to the
/// reactor, because that is the only place a socket option can still be set — after
/// `into_raw_fd` the reactor owns it. `SO_RCVTIMEO` is what turns a server that goes
/// silent mid-body into a completed-with-error read instead of a permanent hang; the
/// same guard `rtsp/src/client.rs` puts on its control connection.
// COLD: one per connection; blocking by design (see `HttpSrc::open`).
#[allow(clippy::disallowed_methods)]
fn connect_socket(url: &Url, timeout: Duration, display: &str) -> Result<TcpStream, Error> {
    use std::net::ToSocketAddrs;
    let addrs = (url.host.as_str(), url.port)
        .to_socket_addrs()
        .map_err(|e| Error::Resource(format!("httpsrc: resolve {} for {display}: {e}", url.host)))?;
    let mut last: Option<std::io::Error> = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, timeout) {
            Ok(stream) => {
                stream.set_read_timeout(Some(timeout)).map_err(|e| {
                    Error::Resource(format!("httpsrc: set receive timeout on {display}: {e}"))
                })?;
                stream.set_write_timeout(Some(timeout)).map_err(|e| {
                    Error::Resource(format!("httpsrc: set send timeout on {display}: {e}"))
                })?;
                return Ok(stream);
            }
            Err(e) => last = Some(e),
        }
    }
    Err(Error::Resource(match last {
        Some(e) => format!("httpsrc: connect {display}: {e}"),
        None => format!("httpsrc: connect {display}: host resolved to no addresses"),
    }))
}

/// Write the request head and read the response head back, over TLS or in the clear.
/// Returns the head plus whether the peer sent `close_notify` in the same breath.
// Blocking write + read: `start()`/reconnect-time setup, the sanctioned exception to
// the workspace blocking-IO lint (the streaming body rides the reactor).
#[allow(clippy::disallowed_methods)]
fn send_and_read_head(
    stream: &mut TcpStream,
    tls: Option<&mut rustls::ClientConnection>,
    request: &str,
    url: &str,
) -> Result<(RawHead, bool), Error> {
    let Some(conn) = tls else {
        stream
            .write_all(request.as_bytes())
            .map_err(|e| Error::Resource(format!("httpsrc: send request to {url}: {e}")))?;
        return Ok((read_head(stream, url)?, false));
    };
    // [`rustls::Stream`] runs the session's state machine over the socket
    // synchronously, so the head is read exactly as it is in the clear.
    let mut s = rustls::Stream::new(conn, stream);
    s.write_all(request.as_bytes())
        .map_err(|e| Error::Resource(format!("httpsrc: send request to {url}: {e}")))?;
    let mut head = read_head(&mut s, url)?;
    // `read_head` stopped at the header terminator, but the TLS record that carried it
    // may have decrypted *more* plaintext than the header read consumed — still
    // buffered in the session. Move it into the leftover now: `process()` only feeds
    // the session ciphertext from new completions and would never see these bytes.
    let state = s
        .conn
        .process_new_packets()
        .map_err(|e| Error::Resource(format!("httpsrc: TLS error from {url}: {e}")))?;
    let plain = state.plaintext_bytes_to_read();
    if plain > 0 {
        let start = head.leftover.len();
        head.leftover.resize(start + plain, 0);
        // In-memory drain of already-decrypted bytes — no fd, no blocking (see
        // `decrypt_into_inbuf`).
        s.conn
            .reader()
            .read_exact(&mut head.leftover[start..])
            .map_err(|e| Error::Resource(format!("httpsrc: TLS read from {url}: {e}")))?;
    }
    Ok((head, state.peer_has_closed()))
}

/// Outcome of locating a full line in buffered input.
enum LineResult {
    /// A complete line is buffered: its length in `self.avail()`, CRLF excluded. The
    /// caller reads those bytes in place and advances the cursor past the line + CRLF.
    Line(usize),
    /// No complete line buffered yet — read more from the socket and retry.
    NeedMore,
}

/// Outcome of a body-fill step (one of `fill_length`/`fill_chunked`/`fill_eof`).
enum FillOutcome {
    /// Wrote `usize` decoded octets into `dst`; `bool` is EOS (body complete).
    Done(usize, bool),
    /// The decoder needs bytes that are not buffered yet and the socket is not at
    /// EOF — `process()` must submit another read and try again on a later pass.
    NeedMore,
    /// The connection cannot deliver the rest of the body: it errored, stalled, or
    /// closed with octets still owed by the framing, and everything that *was*
    /// buffered has already been emitted. `process()` routes this to the resume path
    /// ([`HttpSrc::on_broken`]), which reconnects with a `Range` or fails loudly —
    /// the decision is not the decoder's to make.
    Broken,
}

/// Index of the first `\r\n` in `buf`, if any.
fn find_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\r\n")
}

/// Parse a `chunk-size` (RFC 9112 §7.1: `1*HEXDIG`), ignoring any `chunk-ext` that
/// follows a `;`. The size is a `u64`; overflow of the hex numeral is a protocol error
/// (§7.1 requires recipients to guard against integer conversion overflow).
fn parse_chunk_size(line: &[u8], url: &str) -> Result<u64, Error> {
    // The size is the hex run before an optional `;` chunk-ext (and any BWS).
    let end = line.iter().position(|&b| b == b';').unwrap_or(line.len());
    let hex = &line[..end];
    let hex = trim_ascii(hex);
    if hex.is_empty() {
        return Err(Error::Resource(format!(
            "httpsrc: empty chunk-size from {url}"
        )));
    }
    let mut size: u64 = 0;
    for &b in hex {
        let digit = match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            _ => {
                return Err(Error::Resource(format!(
                    "httpsrc: invalid chunk-size {:?} from {url}",
                    String::from_utf8_lossy(hex)
                )));
            }
        };
        size = size
            .checked_mul(16)
            .and_then(|s| s.checked_add(u64::from(digit)))
            .ok_or_else(|| {
                Error::Resource(format!("httpsrc: chunk-size overflow from {url}"))
            })?;
    }
    Ok(size)
}

/// Trim leading/trailing ASCII whitespace from a byte slice (no `str` allocation).
fn trim_ascii(mut b: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = b {
        if first.is_ascii_whitespace() {
            b = rest;
        } else {
            break;
        }
    }
    while let [rest @ .., last] = b {
        if last.is_ascii_whitespace() {
            b = rest;
        } else {
            break;
        }
    }
    b
}

/// The response head as it came off the wire: the status code, the header block, and
/// any body bytes that arrived in the same read as the header terminator.
///
/// Deliberately *un*interpreted beyond the status line. A redirect's body is thrown
/// away with its connection, so a `302` that arrives `Content-Encoding: gzip` (or with
/// framing this element could not decode) must not fail the chain — only a response we
/// are going to *stream* has to pass [`check_transfer_coding`], [`check_content_coding`]
/// and [`parse_framing`].
struct RawHead {
    /// The three-digit status code (RFC 9112 §4).
    status: u16,
    /// Status line + field lines, without the trailing CRLF CRLF.
    block: Vec<u8>,
    /// Bytes read past `\r\n\r\n` — the start of the body, to feed the decoder.
    leftover: Vec<u8>,
}

/// Read from `stream` until the CRLF-CRLF header terminator and validate the status
/// line. Generic over the transport: a plain [`TcpStream`], or a [`rustls::Stream`]
/// running the TLS session over it.
// Blocking reads are `start()`-time setup (the response head), not element streaming
// IO — the sanctioned exception to the workspace blocking-IO lint.
#[allow(clippy::disallowed_methods)]
fn read_head(stream: &mut impl Read, url: &str) -> Result<RawHead, Error> {
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    loop {
        // A dead-air stall lands here as `WouldBlock`/`TimedOut` (SO_RCVTIMEO), which
        // is what keeps a server that accepts and then says nothing from hanging the
        // group thread forever inside `start()`.
        let n = stream
            .read(&mut chunk)
            .map_err(|e| Error::Resource(format!("httpsrc: read headers from {url}: {e}")))?;
        if n == 0 {
            return Err(Error::Resource(format!(
                "httpsrc: connection closed before end of headers from {url}"
            )));
        }
        buf.extend_from_slice(&chunk[..n]);

        if let Some(pos) = find_header_end(&buf) {
            let body_start = pos + 4; // skip the "\r\n\r\n"
            let leftover = buf[body_start..].to_vec();
            buf.truncate(pos); // keep just the status line + header lines
            let status = parse_status(&buf, url)?;
            return Ok(RawHead { status, block: buf, leftover });
        }

        if buf.len() > MAX_HEADER_BYTES {
            return Err(Error::Resource(format!(
                "httpsrc: response headers exceeded {MAX_HEADER_BYTES} bytes from {url}"
            )));
        }
    }
}

/// Everything a *final* (2xx) response head says that the body path needs. Refuses
/// anything we would have to decode before the body is the resource, in the order the
/// sender wrapped it: transfer coding (a property of the message) outside, content
/// coding (a property of the representation) inside — then decides framing.
fn interpret_head(raw: &RawHead, url: &str) -> Result<Framing, Error> {
    check_transfer_coding(&raw.block, url)?;
    check_content_coding(&raw.block, url)?;
    parse_framing(&raw.block, url)
}

/// Index of the `\r\n\r\n` that separates headers from body, if present.
fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Validate the status line — `HTTP/1.x <code> <reason>` (RFC 9112 §4) — and return
/// the status code. Whether that code is one we can *use* is a separate question, and
/// a different answer for a redirect than for a final response: see
/// [`Disposition::of`].
fn parse_status(header_block: &[u8], url: &str) -> Result<u16, Error> {
    let text = String::from_utf8_lossy(header_block);
    let status_line = text.lines().next().unwrap_or("");
    let mut parts = status_line.split_whitespace();
    let version = parts.next().unwrap_or("");
    if !version.starts_with("HTTP/") {
        return Err(Error::Resource(format!(
            "httpsrc: malformed status line from {url}: {status_line:?}"
        )));
    }
    // §4: the status code is exactly three digits. `parse::<u16>` would also take
    // `+7` and `65535`; neither is a status code, and a "2xx" built out of one is a
    // response we would go on to stream.
    let code = parts.next().unwrap_or("");
    if code.len() != 3 || !code.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Error::Resource(format!(
            "httpsrc: malformed status line from {url}: {status_line:?}"
        )));
    }
    code.parse::<u16>().map_err(|_| {
        Error::Resource(format!(
            "httpsrc: malformed status line from {url}: {status_line:?}"
        ))
    })
}

/// What to do with a response, by status code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Disposition {
    /// A final response: stream its body.
    Final,
    /// Follow its `Location` (RFC 9110 §15.4).
    Redirect,
}

impl Disposition {
    /// Classify a status code, or say why the download stops here.
    ///
    /// The five redirects followed are the ones that mean "the representation you
    /// asked for is over there": `301 Moved Permanently` (§15.4.2), `302 Found`
    /// (§15.4.3), `303 See Other` (§15.4.4), `307 Temporary Redirect` (§15.4.8) and
    /// `308 Permanent Redirect` (§15.4.9). The method stays `GET` across all of them
    /// — §15.4.4 has 303 redirect "to a different resource, one that is intended to
    /// provide an indirect response", retrieved with GET, and 307/308 exist precisely
    /// to *forbid* the method rewriting that 301/302 acquired by custom (§15.4.8:
    /// "the user agent MUST NOT change the request method"). A GET stays a GET either
    /// way, so there is nothing to rewrite.
    ///
    /// `300 Multiple Choices` (§15.4.1) and `305`/`306` are deliberately not followed:
    /// 300 has no single `Location` to take, and 305 was deprecated as a security
    /// hazard (a response that redirects a client through a proxy of the server's
    /// choosing). `304 Not Modified` cannot occur — we send no conditional request
    /// except `If-Range`, whose failure is a `200`, never a `304`.
    fn of(status: u16, url: &str, status_line: &str) -> Result<Self, Error> {
        match status {
            200..=299 => Ok(Self::Final),
            301 | 302 | 303 | 307 | 308 => Ok(Self::Redirect),
            _ => Err(Error::Resource(format!(
                "httpsrc: {url} returned HTTP status {status} ({status_line})"
            ))),
        }
    }
}

/// The status line, for an error message.
fn status_line(header_block: &[u8]) -> Cow<'_, str> {
    let text = String::from_utf8_lossy(header_block);
    match text {
        Cow::Borrowed(s) => Cow::Borrowed(s.lines().next().unwrap_or("")),
        // COLD: only when the head was not UTF-8, i.e. already malformed.
        #[allow(clippy::disallowed_methods)]
        Cow::Owned(s) => Cow::Owned(s.lines().next().unwrap_or("").to_string()),
    }
}

/// Whether the origin advertises byte ranges (RFC 9110 §14.3 `Accept-Ranges`).
///
/// §14.3: the field lists the range units the server supports; `bytes` is the one
/// defined for HTTP (§14.1), and `none` is the explicit "I support none of them".
/// Absent, the field means nothing either way — many origins serve ranges without
/// advertising them, and a CDN edge may add the field the origin omitted — so this is
/// only ever an *optimistic* input: what actually settles it is a `206`.
fn accepts_byte_ranges(header_block: &[u8]) -> bool {
    let lossy = String::from_utf8_lossy(header_block);
    let text = unfold_obs_fold(&lossy);
    let advertised = header_values(&text, "Accept-Ranges").any(|v| {
        v.split(',')
            .map(str::trim)
            .any(|unit| unit.eq_ignore_ascii_case("bytes"))
    });
    advertised
}

/// A parsed `Content-Range: bytes <first>-<last>/<complete-length|*>` (RFC 9110 §14.4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ContentRange {
    /// First byte position of the part, in the representation.
    first: u64,
    /// Last byte position, inclusive.
    last: u64,
    /// Total length of the representation; `None` for the `*` form (§14.4: sent when
    /// the length is unknown at the time the response is generated).
    complete: Option<u64>,
}

/// Parse a `Content-Range` field value (RFC 9110 §14.4). Returns `None` for anything
/// that is not the `bytes` unit in the `<first>-<last>/<len>` form — including the
/// `unsatisfied-range` form (`bytes *​/1234`, only legal on a 416) and any of the
/// absurdities a hostile origin can put there. All arithmetic is checked: the values
/// are `1*DIGIT` with no stated bound, so `parse::<u64>` failing is a normal outcome,
/// not an error case.
fn parse_content_range(value: &str) -> Option<ContentRange> {
    let rest = value.strip_prefix("bytes")?;
    // §14.4's grammar has a single SP after the unit; be tolerant of extra OWS, but
    // require *some* separator so `bytesfoo` cannot parse.
    if !rest.starts_with(|c: char| c.is_ascii_whitespace()) {
        return None;
    }
    let spec = rest.trim();
    let (range, complete) = spec.split_once('/')?;
    let (first, last) = range.split_once('-')?;
    let first: u64 = digits(first)?;
    let last: u64 = digits(last)?;
    // §14.4: "a valid byte-range-resp ... where last-byte-pos is less than
    // first-byte-pos is invalid".
    if last < first {
        return None;
    }
    let complete = match complete.trim() {
        "*" => None,
        n => {
            let n = digits(n)?;
            // The part has to fit inside the whole.
            if last >= n {
                return None;
            }
            Some(n)
        }
    };
    Some(ContentRange { first, last, complete })
}

/// `1*DIGIT` as a `u64`, rejecting signs, whitespace and overflow.
fn digits(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// The response's `ETag`, if it is a **strong** validator (RFC 9110 §8.8.3).
///
/// Only a strong one is any use to us: §14.5 says a client "MUST NOT generate an
/// If-Range header field ... with an entity tag that is marked as weak", because a
/// weak tag only promises semantic equivalence, and splicing byte range N.. of a
/// semantically-equivalent-but-differently-encoded representation onto the first half
/// of another is exactly the corruption `If-Range` exists to prevent. A weak tag
/// (`W/"…"`) is therefore dropped on the floor rather than sent.
// COLD: once per connection; the owned tag is replayed on later resume requests.
#[allow(clippy::disallowed_methods)]
fn strong_etag(header_block: &[u8]) -> Option<String> {
    let lossy = String::from_utf8_lossy(header_block);
    let text = unfold_obs_fold(&lossy);
    let tag = header_value(&text, "ETag")?.trim();
    // §8.8.3: `entity-tag = [ weak ] opaque-tag`, `opaque-tag = DQUOTE *etagc DQUOTE`.
    if !tag.starts_with('"') || !tag.ends_with('"') || tag.len() < 2 {
        return None; // weak (`W/"…"`) or malformed
    }
    // An etagc is VCHAR except DQUOTE, plus obs-text; a tag with an embedded quote is
    // malformed, and one with a CR/LF would be a header injection on the way back out.
    if tag[1..tag.len() - 1].bytes().any(|b| b < 0x21 || b == b'"') {
        return None;
    }
    Some(tag.to_string())
}

/// Replace every obs-fold in a header block with a SP (RFC 9112 §5.2). A field value
/// could historically be continued on the following line by starting that line with a
/// space or horizontal tab (`obs-fold = OWS CRLF RWS`); §5.2 deprecates it but is
/// explicit about what a client owes it: "a user agent that receives an obs-fold in a
/// response message that is not within a `message/http` container MUST replace each
/// received obs-fold with one or more SP octets prior to interpreting the field value".
///
/// This is not cosmetic. Deciding *what is a field line* is the first act of
/// interpreting a field value, and the lookup below tolerates whitespace around a field
/// name, so a continuation left folded reads as a field line in its own right:
/// `X-Note: see\r\n Content-Length: 3` conjures a `Content-Length` nobody sent as a
/// field, and that value goes on to frame the body. Anything able to get text echoed
/// into a response header could shorten a download to a length of its choosing, with no
/// error raised anywhere.
///
/// Returns [`Cow::Borrowed`] untouched when nothing is folded — every well-formed
/// response, since §5.2 also says senders MUST NOT generate a fold.
// COLD: runs once per response, over the header block only; allocates only for the
// deprecated folded form.
#[allow(clippy::disallowed_methods)]
fn unfold_obs_fold(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();
    let folded = |i: usize| bytes[i] == b'\n' && matches!(bytes.get(i + 1), Some(b' ' | b'\t'));
    if !(0..bytes.len()).any(folded) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    for i in (0..bytes.len()).filter(|&i| folded(i)) {
        // Drop the line break (and the CR before it, when present) and leave a SP in
        // its place; the continuation keeps its own leading whitespace, so the fold
        // becomes the "one or more SP octets" §5.2 asks for. Every cut is at an ASCII
        // CR/LF, so the slices stay on UTF-8 boundaries.
        let end = if i > 0 && bytes[i - 1] == b'\r' { i - 1 } else { i };
        out.push_str(&text[copied..end]);
        out.push(' ');
        copied = i + 1;
    }
    out.push_str(&text[copied..]);
    Cow::Owned(out)
}

/// Every value of header `name` (case-insensitive, RFC 9112 header names are
/// case-insensitive tokens), in the order the field lines appear, each trimmed of
/// surrounding whitespace. `header_block` is the status line plus header lines, without
/// the trailing CRLF-CRLF.
///
/// All of them, not just one: RFC 9110 §5.3 makes repeated field lines of the same name
/// equivalent to a single field whose value is those values joined by commas, so a
/// recipient that reads one line has seen only part of the field. Fields whose framing
/// consequences depend on the whole list (`Content-Length`) must iterate this.
fn header_values<'a>(header_block: &'a str, name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
    header_block
        .lines()
        .skip(1) // the status line, not a header field
        .filter_map(|line| line.split_once(':'))
        .filter(move |(k, _)| k.trim().eq_ignore_ascii_case(name))
        .map(|(_, v)| v.trim())
}

/// Return the last value of header `name`. Later occurrences win, which matters for
/// none of the fields read this way but keeps the lookup unsurprising; see
/// [`header_values`] for the fields where every occurrence has to be considered.
fn header_value<'a>(header_block: &'a str, name: &'a str) -> Option<&'a str> {
    header_values(header_block, name).last()
}

/// Reject a transfer-coding stack we cannot fully undo (RFC 9112 §6.1
/// `Transfer-Encoding`).
///
/// §6.1: the field "lists the transfer coding names corresponding to the sequence of
/// transfer codings that have been (or will be) applied to the content in order to form
/// the message body" — a list the recipient has to unwind in full. Its own example is
/// `gzip, chunked`.
///
/// This is deliberately a different question from the one [`parse_framing`] answers.
/// *Which* coding is final decides the framing (§6.3 rule 4), and for `gzip, chunked`
/// that is `chunked`, correctly — but undoing that outer layer only uncovers the gzip
/// member underneath, and `chunked` (§7.1) is the sole coding this element implements.
/// Emitting what falls out is the same silent corruption as an undecoded content coding
/// (see [`check_content_coding`]), so framing selection stays where it is and the
/// decodability of the rest of the stack is checked here.
///
/// §6.1 phrases the obligation from the sending side — a sender may only stack codings
/// under a final `chunked` (or close-delimit the response), and a server facing a coding
/// it does not understand "SHOULD respond with 501 (Not Implemented)". A *response*
/// recipient has no such reply to send, which leaves failing loudly as the only honest
/// move: the alternative is handing coded bytes downstream as the payload.
fn check_transfer_coding(header_block: &[u8], url: &str) -> Result<(), Error> {
    // Same lossy-then-unfold reading as `parse_framing` (§5.2), and every field line,
    // since repeated lines are one comma-joined list (RFC 9110 §5.3).
    let lossy = String::from_utf8_lossy(header_block);
    let text = unfold_obs_fold(&lossy);
    let mut codings = 0usize;
    for value in header_values(&text, "Transfer-Encoding") {
        for coding in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            codings += 1;
            // §7: transfer-coding names are case-insensitive tokens.
            if !coding.eq_ignore_ascii_case("chunked") {
                return Err(Error::Resource(format!(
                    "httpsrc: {url} applied Transfer-Encoding {coding:?} — this element \
                     decodes only `chunked` and will not pass coded bytes off as the body"
                )));
            }
        }
    }
    // §6.1: "A sender MUST NOT apply the chunked transfer coding more than once to a
    // message body." `fill_chunked` undoes exactly one layer, so a repeat would leave
    // chunk framing sitting in the bytes we emit.
    if codings > 1 {
        return Err(Error::Resource(format!(
            "httpsrc: {url} applied the chunked transfer coding {codings} times \
             (RFC 9112 §6.1: at most once); this element decodes one layer"
        )));
    }
    Ok(())
}

/// Reject a response whose body is not the resource yet (RFC 9110 §8.4
/// `Content-Encoding`).
///
/// A *transfer* coding is framing, and this element strips the one it supports. A
/// *content* coding is part of the representation itself — §8.4 defines the field as
/// "what decoding mechanisms have to be applied in order to obtain data in the media
/// type referenced by the Content-Type header field" — so it outlives de-chunking:
/// decode `Transfer-Encoding: chunked` off a `Content-Encoding: gzip` response and what
/// falls out is the gzip member, not the file. Pushing that downstream as the payload
/// is silent corruption, handing a demuxer DEFLATE noise where a container should be.
/// We implement no content decoders, so the only honest move is to say so and stop.
///
/// The request advertises `Accept-Encoding: identity`, which is what makes a coded
/// response the server's mistake rather than ours: RFC 9110 §12.5.3 rule 1 is that
/// *without* an `Accept-Encoding` field "any content coding is considered acceptable by
/// the user agent", so a client with no decoders has to ask for none.
fn check_content_coding(header_block: &[u8], url: &str) -> Result<(), Error> {
    // Same lossy-then-unfold reading as `parse_framing` — a folded `Content-Encoding`
    // must not read as an empty one and slip through.
    let lossy = String::from_utf8_lossy(header_block);
    let text = unfold_obs_fold(&lossy);
    for value in header_values(&text, "Content-Encoding") {
        // §8.4: a list of codings in the order they were applied; the names are
        // case-insensitive tokens (RFC 9112 §7).
        for coding in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            // §8.4.1: `identity` is the reserved "no transformation" name — a sender
            // SHOULD NOT even list it, but it is nothing we would have to undo.
            if !coding.eq_ignore_ascii_case("identity") {
                return Err(Error::Resource(format!(
                    "httpsrc: {url} applied Content-Encoding {coding:?} despite \
                     `Accept-Encoding: identity` — this element implements no content \
                     decoders and will not pass coded bytes off as the body"
                )));
            }
        }
    }
    Ok(())
}

/// Parse a `Content-Length` value, which RFC 9112 §6.2 spells `1*DIGIT`. `str::parse`
/// alone is too generous: it also accepts a leading `+`, so `Content-Length: +5` would
/// slip through as 5 and frame a longer body short — the same silent truncation, one
/// octet away. A sign is also precisely the kind of difference that makes two
/// recipients on a path disagree about where a body ends (§11.2).
fn parse_content_length(value: &str) -> Option<u64> {
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    value.parse().ok() // still fallible: more digits than fit a u64
}

/// Choose the message-body framing from the response header block (RFC 9112 §6.3).
///
/// Precedence: `Transfer-Encoding` overrides `Content-Length` (§6.3 rule 3); a valid
/// `Content-Length` frames the body otherwise (§6.2); with neither, the body is read
/// to EOF (§6.3 rule 8). Only the `chunked` transfer coding — and only as the final
/// coding — is decoded; any other coding is rejected, since we implement no content
/// decoders.
fn parse_framing(header_block: &[u8], url: &str) -> Result<Framing, Error> {
    // Field values are not required to be UTF-8, so the block is read lossily. Any
    // obs-fold is unfolded first: §5.2 requires it before a field value is interpreted,
    // and which lines *are* field lines is the first thing framing depends on.
    let lossy = String::from_utf8_lossy(header_block);
    let text = unfold_obs_fold(&lossy);

    // RFC 9112 §6.1: Transfer-Encoding is a comma-separated list of coding names; the
    // *final* one determines framing (§6.3 rule 4). Names are case-insensitive (§7).
    if let Some(te) = header_value(&text, "Transfer-Encoding") {
        let final_coding = te
            .rsplit(',')
            .map(str::trim)
            .find(|s| !s.is_empty())
            .unwrap_or("");
        if final_coding.eq_ignore_ascii_case("chunked") {
            return Ok(Framing::Chunked(ChunkState::Size));
        }
        return Err(Error::Resource(format!(
            "httpsrc: unsupported Transfer-Encoding {te:?} from {url} \
             (only chunked is decoded)"
        )));
    }

    // RFC 9112 §6.2: a decimal octet count. §6.3 rule 5 allows a list only if every
    // value is valid and they are all identical; anything else makes the framing
    // invalid, and for a response "received by a user agent, the user agent MUST close
    // the connection to the server and discard the received response".
    //
    // That list arrives in either of two spellings — comma-separated inside one field
    // line, or spread over repeated `Content-Length` field lines, which RFC 9110 §5.3
    // defines as the same field — so both are folded into one pass here. Reading only
    // one of the lines is how `Content-Length: 100` followed by `Content-Length: 3`
    // frames a 100-byte body as 3 bytes and reports a successful download: a short file
    // the caller has no way to notice.
    let mut length: Option<u64> = None;
    for value in header_values(&text, "Content-Length") {
        for part in value.split(',').map(str::trim) {
            let n = parse_content_length(part).ok_or_else(|| {
                Error::Resource(format!(
                    "httpsrc: invalid Content-Length {part:?} from {url}"
                ))
            })?;
            match length {
                Some(prev) if prev != n => {
                    return Err(Error::Resource(format!(
                        "httpsrc: conflicting Content-Length ({prev} then {n}) from {url}"
                    )));
                }
                _ => length = Some(n),
            }
        }
    }
    if let Some(len) = length {
        return Ok(Framing::Length(len));
    }

    // RFC 9112 §6.3 rule 8: no declared length → close-delimited body.
    Ok(Framing::Eof)
}

impl Element for HttpSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    // One-time synchronous setup (connect, TLS handshake, request write, header
    // read) — the sanctioned exception to the workspace blocking-IO lint; the
    // streaming body rides the reactor. The clone is the one-time reset of the
    // effective URL, not a per-buffer allocation.
    #[allow(clippy::disallowed_methods)]
    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        // A restart re-resolves everything: the redirect chain may have moved, and the
        // validator/length belong to the connection they came from.
        self.effective_url = self.url.clone();
        self.cursor = 0;
        self.done = false;
        self.in_flight = 0;
        self.io_err = None;
        self.socket_eof = false;
        self.seq = 0;
        self.valid_from = 0;
        self.pos = 0;
        self.total_len = None;
        self.ranges_ok = false;
        self.etag = None;
        self.pending = None;
        self.retries = 0;
        self.discont = false;
        self.tls_closed = false;
        self.tls = None;
        self.publish_info();

        // One-time setup stays synchronous (spec: IO — connect + request write are a
        // `start()`-time act, like `filesrc` opening its file): connect, TLS handshake
        // for `https` ([`crate::tls`]), GET, follow any redirects, read the response
        // head. Only the streaming *body* reads ride the reactor.
        let live = self.open(ctx, None)?;
        self.install(ctx, live, 0, false)
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.done {
            return Ok(Flow::Eos);
        }

        // A (re)connect owns the pass: nothing can be decoded out of a connection that
        // no longer exists, and the socket the reactor holds is about to be replaced.
        if self.pending.is_some() {
            if !self.run_pending(ctx)? {
                return Ok(Flow::Ok);
            }
            if self.done {
                return Ok(Flow::Eos); // a seek past the end of the resource
            }
        }

        // Drain any completed body read into `inbuf` (frees the read-in-flight slot).
        self.drain_reads(ctx)?;
        // A socket error means no further bytes will arrive on this connection; the
        // decoder must be told so it stops asking for more, and it drains whatever is
        // buffered before the resume path takes over.
        if self.io_err.is_some() {
            self.socket_eof = true;
        }

        // Grab a pooled output buffer; `None` is pool backpressure — but still keep a
        // read topped up so bytes are on the way when a slot frees. Retry next pass.
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => {
                self.submit_read(ctx);
                return Ok(Flow::Ok);
            }
        };

        // Fill the buffer with decoded body bytes according to the framing.
        let cap = buf.memory.capacity();
        let dst = buf.memory.as_mut_full();
        let outcome = match self.framing {
            Framing::Length(_) => self.fill_length(&mut dst[..cap])?,
            Framing::Chunked(_) => self.fill_chunked(&mut dst[..cap])?,
            Framing::Eof => self.fill_eof(&mut dst[..cap])?,
        };

        let (written, eos) = match outcome {
            FillOutcome::Done(w, e) => (w, e),
            FillOutcome::NeedMore => {
                // Decoder is starved: submit another socket read and come back. The
                // pooled `buf` recycles on drop (nothing was written into it).
                self.submit_read(ctx);
                return Ok(Flow::Ok);
            }
            FillOutcome::Broken => {
                // The connection died with body still owed. Resume it from `pos` (the
                // pooled `buf` recycles unused), or fail with a message that says how
                // far we got and why it cannot go further.
                drop(buf);
                self.on_broken(ctx)?;
                return Ok(Flow::Ok);
            }
        };

        if eos {
            self.done = true;
        } else {
            // Keep one read in flight so the next `process()` finds fresh bytes rather
            // than starting the read then (single-op streaming — see the module docs).
            self.submit_read(ctx);
        }

        // Push non-empty buffers; skip empty ones so we don't emit a zero-length
        // buffer just to carry EOS (e.g. an empty chunked step that only consumed
        // framing bytes). `try_alloc` again next call is cheap.
        if written > 0 {
            // The byte accounting, in one place: everything pushed is what `pos`
            // counts, so a resume asks for exactly the bytes downstream has not seen,
            // and a seek's `Range` start is the offset the next push carries.
            self.pos += written as u64;
            buf.memory.set_len(written);
            if self.discont {
                // First buffer after a seek: the byte stream jumped (spec: Buffer —
                // DISCONT). A resume is contiguous and sets nothing.
                self.discont = false;
                buf.flags |= BufferFlags::DISCONT;
            }
            ctx.out(PadId(0)).push(buf);
        }

        Ok(if self.done { Flow::Eos } else { Flow::Ok })
    }

    fn event(&mut self, ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        if !matches!(event, Event::FlushStart) {
            return Ok(());
        }
        // Note what is deliberately *not* done here: the `seq` floor is not raised.
        // `filesrc` raises it on every flush because its reads are positioned, so a
        // completion submitted before a seek carries bytes from the wrong offset. A
        // socket read has no offset — its bytes are simply the next ones on this
        // connection, and they stay correct unless the connection itself is replaced.
        // Discarding them here instead punched a hole in the byte stream on any flush
        // this element did not act on (a seek it refuses, or one that lands where it
        // already is): the bytes were gone from the socket and never emitted. The floor
        // is raised in `install`, which is the only place a connection is replaced.
        let Some(target) = ctx.seek_target() else {
            return Ok(()); // a flush with no seek behind it: nothing to reposition
        };
        let at = target.to_byte;

        // Already streaming from exactly there (a seek that landed on the current
        // position, or a repeat of one already scheduled): the bytes about to be
        // pushed are the ones being asked for, so tearing the connection down and
        // rebuilding it would only cost a round trip.
        if at == self.pos && self.pending.is_none() && self.file.is_some() && !self.done {
            self.discont = true;
            return Ok(());
        }

        // Seeking to or past the end is EOS, not a request. Asking for it would be an
        // unsatisfiable range — RFC 9110 §14.1.2: a byte-range-spec whose
        // first-byte-pos is greater than the current length of the representation is
        // unsatisfiable, and §15.5.17 answers that with `416`.
        if self.total_len.is_some_and(|total| at >= total) {
            self.pos = self.total_len.unwrap_or(at);
            self.pending = None;
            self.done = true;
            return Ok(());
        }

        // Anywhere but byte zero needs ranges *and* a known length (see `seekable`).
        // Refusing the target is not an error: an un-seekable stream is a perfectly
        // good stream, and killing the pipeline over a seek the app was told (via
        // `HttpInfo::is_seekable`) would not work is the wrong trade. Say so on the
        // bus and keep streaming.
        if at != 0 && !self.seekable() {
            let msg = format!(
                "httpsrc: {} ignoring seek to byte {at}: {} — streaming on from byte {}",
                self.effective_url,
                if self.ranges_ok {
                    "the resource length is unknown (no Content-Length)"
                } else {
                    "the server does not serve byte ranges (RFC 9110 §14.3)"
                },
                self.pos
            );
            self.warn(ctx, msg);
            return Ok(());
        }

        // Reopen at the target. The connection is torn down and rebuilt by
        // `run_pending` on the next `process()` — never here, so a blocking connect
        // does not run inside the scheduler's event delivery, where every element in
        // the group is waiting on it.
        self.pending = Some(Pending {
            at,
            not_before: None,
        });
        // A seek is a user act, not a failure: it does not spend the retry budget.
        self.retries = 0;
        self.done = false;
        self.discont = true;
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // The reactor owns and closes the socket fd (registered in `start`). Just drop
        // our handle and buffered state. Dropping the TLS session without close_notify
        // is fine: we are the reader, with nothing left to authenticate to the peer.
        self.file = None;
        self.inbuf.clear();
        self.cursor = 0;
        self.in_flight = 0;
        self.io_err = None;
        self.pending = None;
        self.socket_eof = false;
        self.tls = None;
        self.tls_closed = false;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        accepts_byte_ranges, check_content_coding, check_transfer_coding, header_value,
        header_values, parse_chunk_size, parse_content_range, parse_framing, parse_http_url,
        parse_status, remove_dot_segments, resolve_reference, strong_etag, unfold_obs_fold,
        ContentRange, Cow, Disposition, Framing,
    };

    #[test]
    fn parses_full_url() {
        let u = parse_http_url("http://example.com:8080/path/to/file").unwrap();
        assert_eq!(u.host, "example.com");
        assert_eq!(u.port, 8080);
        assert_eq!(u.path, "/path/to/file");
    }

    #[test]
    fn defaults_port_and_path() {
        let u = parse_http_url("http://example.com").unwrap();
        assert_eq!(u.host, "example.com");
        assert_eq!(u.port, 80);
        assert_eq!(u.path, "/");
    }

    #[test]
    fn ipv4_with_port() {
        let u = parse_http_url("http://127.0.0.1:54321/file").unwrap();
        assert_eq!(u.host, "127.0.0.1");
        assert_eq!(u.port, 54321);
        assert_eq!(u.path, "/file");
    }

    #[test]
    fn https_default_port_and_flag() {
        let u = parse_http_url("https://example.com/file").unwrap();
        assert!(u.tls);
        assert_eq!(u.host, "example.com");
        assert_eq!(u.port, 443);
        assert_eq!(u.path, "/file");

        let u = parse_http_url("https://example.com:8443/").unwrap();
        assert!(u.tls);
        assert_eq!(u.port, 8443);

        let u = parse_http_url("http://example.com/").unwrap();
        assert!(!u.tls, "plain http is not TLS");
    }

    #[test]
    fn rejects_other_schemes() {
        assert!(parse_http_url("ftp://example.com/").is_err());
        assert!(parse_http_url("example.com/").is_err());
    }

    #[test]
    fn rejects_bad_port() {
        assert!(parse_http_url("http://example.com:notaport/").is_err());
    }

    #[test]
    fn status_line_parses_the_code_and_rejects_malformed() {
        assert_eq!(parse_status(b"HTTP/1.1 200 OK", "u").unwrap(), 200);
        assert_eq!(parse_status(b"HTTP/1.1 204 No Content", "u").unwrap(), 204);
        assert_eq!(parse_status(b"HTTP/1.1 404 Not Found", "u").unwrap(), 404);
        assert!(parse_status(b"garbage", "u").is_err());
        assert!(parse_status(b"HTTP/1.1", "u").is_err());
        // RFC 9112 §4: exactly three digits — no sign, no short/long numeral.
        assert!(parse_status(b"HTTP/1.1 +20 OK", "u").is_err());
        assert!(parse_status(b"HTTP/1.1 20 OK", "u").is_err());
        assert!(parse_status(b"HTTP/1.1 2000 OK", "u").is_err());
    }

    #[test]
    fn disposition_follows_the_five_redirects_and_refuses_the_rest() {
        let d = |code| Disposition::of(code, "u", "line");
        assert_eq!(d(200).unwrap(), Disposition::Final);
        assert_eq!(d(206).unwrap(), Disposition::Final);
        assert_eq!(d(204).unwrap(), Disposition::Final);
        // RFC 9110 §15.4.2/.3/.4/.8/.9 — GET stays GET across all of them.
        for code in [301u16, 302, 303, 307, 308] {
            assert_eq!(d(code).unwrap(), Disposition::Redirect, "status {code}");
        }
        // 300 has no single Location; 305 was deprecated as a security hazard.
        assert!(d(300).is_err());
        assert!(d(305).is_err());
        assert!(d(404).is_err());
        assert!(d(500).is_err());
    }

    #[test]
    fn reference_resolution_matches_rfc3986_examples() {
        // RFC 3986 §5.4.1 "Normal Examples", verbatim, against §5.4's base URI.
        let base = "http://a/b/c/d;p?q";
        let cases = [
            ("g:h", None), // a different scheme: not something we follow
            ("g", Some("http://a/b/c/g")),
            ("./g", Some("http://a/b/c/g")),
            ("g/", Some("http://a/b/c/g/")),
            ("/g", Some("http://a/g")),
            // §5.4.1's answer is `http://g`; we recompose the empty path as `/`, which
            // RFC 9110 §4.2.3 defines as the same origin-form request target.
            ("//g", Some("http://g/")),
            ("?y", Some("http://a/b/c/d;p?y")),
            ("g?y", Some("http://a/b/c/g?y")),
            ("#s", None), // fragment-only: nothing to request
            ("g#s", Some("http://a/b/c/g")),
            ("g?y#s", Some("http://a/b/c/g?y")),
            (";x", Some("http://a/b/c/;x")),
            ("g;x", Some("http://a/b/c/g;x")),
            ("g;x?y#s", Some("http://a/b/c/g;x?y")),
            ("./", Some("http://a/b/c/")),
            ("../", Some("http://a/b/")),
            ("../g", Some("http://a/b/g")),
            ("../../", Some("http://a/")),
            ("../../g", Some("http://a/g")),
        ];
        for (reference, want) in cases {
            match (resolve_reference(base, reference), want) {
                (Ok(got), Some(want)) => assert_eq!(got, want, "resolving {reference:?}"),
                (Err(_), None) => {}
                (got, want) => panic!("resolving {reference:?}: got {got:?}, wanted {want:?}"),
            }
        }
    }

    #[test]
    fn reference_resolution_abnormal_cases_stay_total() {
        // RFC 3986 §5.4.2 "Abnormal Examples": walking above the root is bounded, and
        // the dot-segment rules apply to complete segments only.
        let base = "http://a/b/c/d;p?q";
        for (reference, want) in [
            ("../../../g", "http://a/g"),
            ("../../../../g", "http://a/g"),
            ("/./g", "http://a/g"),
            ("/../g", "http://a/g"),
            ("g.", "http://a/b/c/g."),
            (".g", "http://a/b/c/.g"),
            ("g..", "http://a/b/c/g.."),
            ("..g", "http://a/b/c/..g"),
            ("./../g", "http://a/b/g"),
            ("./g/.", "http://a/b/c/g/"),
            ("g/./h", "http://a/b/c/g/h"),
            ("g/../h", "http://a/b/c/h"),
            ("g;x=1/./y", "http://a/b/c/g;x=1/y"),
            ("g;x=1/../y", "http://a/b/c/y"),
        ] {
            assert_eq!(
                resolve_reference(base, reference).unwrap(),
                want,
                "resolving {reference:?}"
            );
        }
    }

    #[test]
    fn reference_resolution_keeps_scheme_host_and_port() {
        // A network-path reference inherits the scheme (§4.2) — the way a site moves a
        // redirect between http and https without naming either.
        assert_eq!(
            resolve_reference("https://a/x", "//b/y").unwrap(),
            "https://b/y"
        );
        // A nonstandard port survives into the recomposed URL (§3.2.3), and the
        // default one is not re-added.
        assert_eq!(
            resolve_reference("http://a:8080/b/c", "d").unwrap(),
            "http://a:8080/b/d"
        );
        assert_eq!(
            resolve_reference("http://a:80/b/c", "d").unwrap(),
            "http://a/b/d"
        );
        // Scheme changes in both directions are resolvable (the *warning* about a
        // downgrade is `open`'s job, not the resolver's).
        assert_eq!(
            resolve_reference("https://a/x", "http://b/y").unwrap(),
            "http://b/y"
        );
        // A base with no path at all still merges to an absolute path (§5.2.3).
        assert_eq!(resolve_reference("http://a", "g").unwrap(), "http://a/g");
    }

    #[test]
    fn reference_resolution_rejects_hostile_locations() {
        let base = "http://a/b/c";
        assert!(resolve_reference(base, "").is_err());
        assert!(resolve_reference(base, "   ").is_err());
        assert!(resolve_reference(base, "#frag").is_err());
        // Schemes we do not speak, including the ones that make a redirect a weapon.
        assert!(resolve_reference(base, "ftp://h/x").is_err());
        assert!(resolve_reference(base, "javascript:alert(1)").is_err());
        assert!(resolve_reference(base, "file:///etc/passwd").is_err());
        // Header injection through the redirect (RFC 9112 §11.1) — caught by the same
        // octet check the constructor URL goes through.
        assert!(resolve_reference(base, "/x\r\nX-Injected: 1").is_err());
        assert!(resolve_reference(base, "http://h/x\r\nX: 1").is_err());
        // No host, bad port.
        assert!(resolve_reference(base, "http:///x").is_err());
        assert!(resolve_reference(base, "http://h:notaport/x").is_err());
        // Absurd length.
        let long = format!("/{}", "a".repeat(super::MAX_URL_BYTES));
        assert!(resolve_reference(base, &long).is_err());
    }

    #[test]
    fn dot_segment_removal_is_total() {
        // RFC 3986 §5.2.4's own worked example.
        assert_eq!(remove_dot_segments("/a/b/c/./../../g"), "/a/g");
        assert_eq!(remove_dot_segments("mid/content=5/../6"), "mid/6");
        // Nothing to remove.
        assert_eq!(remove_dot_segments("/a/b"), "/a/b");
        assert_eq!(remove_dot_segments(""), "");
        // Walking above the root stops at the root rather than escaping it.
        assert_eq!(remove_dot_segments("/../../../x"), "/x");
        assert_eq!(remove_dot_segments("/.."), "/");
        assert_eq!(remove_dot_segments("/."), "/");
        assert_eq!(remove_dot_segments(".."), "");
        assert_eq!(remove_dot_segments("."), "");
        assert_eq!(remove_dot_segments("/a/.."), "/");
        assert_eq!(remove_dot_segments("/a/."), "/a/");
    }

    #[test]
    fn accept_ranges_is_read_per_rfc9110_14_3() {
        let yes = b"HTTP/1.1 200 OK\r\nAccept-Ranges: bytes";
        assert!(accepts_byte_ranges(yes));
        // Case-insensitive token, and one entry of a list is enough.
        assert!(accepts_byte_ranges(b"HTTP/1.1 200 OK\r\naccept-ranges: BYTES"));
        assert!(accepts_byte_ranges(
            b"HTTP/1.1 200 OK\r\nAccept-Ranges: none, bytes"
        ));
        // §14.3: `none` is the explicit "no range unit supported".
        assert!(!accepts_byte_ranges(b"HTTP/1.1 200 OK\r\nAccept-Ranges: none"));
        // Absent says nothing either way — and must not read as a yes.
        assert!(!accepts_byte_ranges(b"HTTP/1.1 200 OK\r\nContent-Length: 5"));
    }

    #[test]
    fn content_range_parses_and_refuses_nonsense() {
        // RFC 9110 §14.4: `bytes <first>-<last>/<complete-length|*>`.
        assert_eq!(
            parse_content_range("bytes 100-199/1234"),
            Some(ContentRange { first: 100, last: 199, complete: Some(1234) })
        );
        assert_eq!(
            parse_content_range("bytes 0-0/1"),
            Some(ContentRange { first: 0, last: 0, complete: Some(1) })
        );
        // The `*` complete-length form: the total was not known when the response was
        // generated. Legal, and it means we still cannot seek.
        assert_eq!(
            parse_content_range("bytes 5-9/*"),
            Some(ContentRange { first: 5, last: 9, complete: None })
        );
        // Nonsense a hostile origin can put there — every one of these must be `None`,
        // never a panic and never a range we would splice bytes onto.
        for bad in [
            "",
            "bytes",
            "bytesfoo 0-1/2",
            "items 0-1/2",       // a range unit we do not implement (§14.1)
            "bytes */1234",      // the unsatisfied-range form (416 only)
            "bytes 5-1/10",      // last < first
            "bytes 0-10/5",      // the part does not fit the whole
            "bytes -1-5/10",
            "bytes 0-5/-10",
            "bytes 0-5",
            "bytes 0/5",
            "bytes a-b/c",
            "bytes 99999999999999999999-99999999999999999999/99999999999999999999",
        ] {
            assert_eq!(parse_content_range(bad), None, "{bad:?} must not parse");
        }
    }

    #[test]
    fn only_strong_etags_are_replayed_as_if_range() {
        // RFC 9110 §8.8.3 opaque-tag, §14.5 "MUST NOT ... with a weak entity tag".
        assert_eq!(
            strong_etag(b"HTTP/1.1 200 OK\r\nETag: \"abc123\"").as_deref(),
            Some("\"abc123\"")
        );
        assert_eq!(strong_etag(b"HTTP/1.1 200 OK\r\nETag: W/\"abc\""), None);
        assert_eq!(strong_etag(b"HTTP/1.1 200 OK\r\nETag: abc"), None);
        // An empty opaque-tag is odd but legal (§8.8.3's `*etagc`), and harmless.
        assert_eq!(
            strong_etag(b"HTTP/1.1 200 OK\r\nETag: \"\"").as_deref(),
            Some("\"\"")
        );
        assert_eq!(strong_etag(b"HTTP/1.1 200 OK\r\nContent-Length: 1"), None);
        // A tag that would inject a field line on the way back out in `If-Range`.
        assert_eq!(strong_etag(b"HTTP/1.1 200 OK\r\nETag: \"a\"b\""), None);
    }

    #[test]
    fn header_lookup_is_case_insensitive_and_trims() {
        let block = "HTTP/1.1 200 OK\r\nContent-Length:  42 \r\nHost: x";
        assert_eq!(header_value(block, "content-length"), Some("42"));
        assert_eq!(header_value(block, "CONTENT-LENGTH"), Some("42"));
        assert_eq!(header_value(block, "host"), Some("x"));
        assert_eq!(header_value(block, "missing"), None);
        // The status line is never treated as a header field.
        assert_eq!(header_value(block, "HTTP/1.1"), None);
    }

    #[test]
    fn framing_content_length() {
        let block = b"HTTP/1.1 200 OK\r\nContent-Length: 1234";
        match parse_framing(block, "u").unwrap() {
            Framing::Length(n) => assert_eq!(n, 1234),
            other => panic!("expected Length, got {other:?}"),
        }
    }

    #[test]
    fn framing_chunked_takes_precedence_over_length() {
        // RFC 9112 §6.3 rule 3: Transfer-Encoding overrides Content-Length.
        let block = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nTransfer-Encoding: chunked";
        assert!(matches!(
            parse_framing(block, "u").unwrap(),
            Framing::Chunked(_)
        ));
    }

    #[test]
    fn framing_chunked_final_coding_and_case() {
        // §6.3 rule 4: only the *final* coding matters; §7 names are case-insensitive.
        let block = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip, Chunked";
        assert!(matches!(
            parse_framing(block, "u").unwrap(),
            Framing::Chunked(_)
        ));
    }

    #[test]
    fn framing_non_chunked_te_is_rejected() {
        let block = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip";
        assert!(parse_framing(block, "u").is_err());
    }

    #[test]
    fn framing_absent_is_eof() {
        let block = b"HTTP/1.1 200 OK\r\nHost: x";
        assert!(matches!(parse_framing(block, "u").unwrap(), Framing::Eof));
    }

    #[test]
    fn framing_bad_and_conflicting_content_length() {
        assert!(parse_framing(b"HTTP/1.1 200 OK\r\nContent-Length: abc", "u").is_err());
        // §6.3 rule 5: a list is only accepted if all values are identical.
        assert!(parse_framing(b"HTTP/1.1 200 OK\r\nContent-Length: 3, 4", "u").is_err());
        match parse_framing(b"HTTP/1.1 200 OK\r\nContent-Length: 7, 7", "u").unwrap() {
            Framing::Length(n) => assert_eq!(n, 7),
            other => panic!("expected Length, got {other:?}"),
        }
    }

    #[test]
    fn chunk_size_parses_hex_and_ignores_ext() {
        assert_eq!(parse_chunk_size(b"a", "u").unwrap(), 10);
        assert_eq!(parse_chunk_size(b"1f", "u").unwrap(), 31);
        assert_eq!(parse_chunk_size(b"FF", "u").unwrap(), 255);
        assert_eq!(parse_chunk_size(b"0", "u").unwrap(), 0);
        // chunk-ext after the size is ignored (RFC 9112 §7.1.1).
        assert_eq!(parse_chunk_size(b"10;name=value", "u").unwrap(), 16);
        assert_eq!(parse_chunk_size(b"a0 ; ext", "u").unwrap(), 160);
    }

    #[test]
    fn chunk_size_rejects_bad_and_overflow() {
        assert!(parse_chunk_size(b"", "u").is_err());
        assert!(parse_chunk_size(b"xyz", "u").is_err());
        assert!(parse_chunk_size(b";only-ext", "u").is_err());
        // 17 hex digits overflow u64 (RFC 9112 §7.1: guard integer overflow).
        assert!(parse_chunk_size(b"1ffffffffffffffff", "u").is_err());
    }

    #[test]
    fn header_lookup_yields_every_field_line() {
        // RFC 9110 §5.3: repeated field lines are one comma-joined field, so a lookup
        // that has to weigh the whole field must see all of them.
        let block = "HTTP/1.1 200 OK\r\nContent-Length: 100\r\nX: y\r\nContent-Length: 3";
        let all: Vec<_> = header_values(block, "content-length").collect();
        assert_eq!(all, ["100", "3"]);
        // ... while the single-value lookup still reports the last occurrence.
        assert_eq!(header_value(block, "Content-Length"), Some("3"));
    }

    #[test]
    fn obs_fold_is_replaced_by_a_space() {
        // RFC 9112 §5.2: a user agent MUST replace each obs-fold with SP before
        // interpreting the field value. Untouched when nothing is folded.
        let plain = "HTTP/1.1 200 OK\r\nContent-Length: 5";
        assert!(matches!(unfold_obs_fold(plain), Cow::Borrowed(_)));
        assert_eq!(unfold_obs_fold("A: one\r\n two"), "A: one  two");
        assert_eq!(unfold_obs_fold("A: one\r\n\ttwo"), "A: one \ttwo");
        // A bare-LF line ending folds too: `.lines()` breaks on it either way.
        assert_eq!(unfold_obs_fold("A: one\n two"), "A: one  two");
        // The last line has no fold to join, and a following field line is not one.
        assert_eq!(unfold_obs_fold("A: one\r\nB: two"), "A: one\r\nB: two");
    }

    #[test]
    fn framing_ignores_a_content_length_smuggled_in_a_folded_value() {
        // The continuation belongs to `X-Note`; unfolded it can no longer pass for a
        // field line of its own, so the real Content-Length stands (§5.2).
        let block = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nX-Note: see\r\n Content-Length: 3";
        match parse_framing(block, "u").unwrap() {
            Framing::Length(n) => assert_eq!(n, 100),
            other => panic!("expected Length(100), got {other:?}"),
        }
    }

    #[test]
    fn framing_rejects_content_length_split_over_field_lines() {
        // §6.3 rule 5 via §5.3: repeated field lines are the same list as one written
        // with commas, so differing values are just as unrecoverable. Taking the last
        // framed a 100-byte body as 3 bytes and called it a completed download.
        let split = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nContent-Length: 3";
        assert!(parse_framing(split, "u").is_err());
        // Identical repeats stay legal, exactly as the comma-list form does.
        let same = b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nContent-Length: 7";
        match parse_framing(same, "u").unwrap() {
            Framing::Length(n) => assert_eq!(n, 7),
            other => panic!("expected Length(7), got {other:?}"),
        }
    }

    #[test]
    fn framing_rejects_a_signed_content_length() {
        // §6.2 is `1*DIGIT`; `str::parse::<u64>` also takes a leading `+`, which framed
        // a 100-byte body as 5 bytes and reported the download complete.
        assert!(parse_framing(b"HTTP/1.1 200 OK\r\nContent-Length: +5", "u").is_err());
        assert!(parse_framing(b"HTTP/1.1 200 OK\r\nContent-Length: 0x10", "u").is_err());
        // A count that does not fit a u64 is invalid too, not a wrapped one.
        let huge = b"HTTP/1.1 200 OK\r\nContent-Length: 99999999999999999999";
        assert!(parse_framing(huge, "u").is_err());
    }

    #[test]
    fn transfer_coding_rejects_only_what_we_cannot_undo() {
        // `chunked` alone is the one coding `fill_chunked` undoes — the happy path, and
        // an absent field is nothing to undo at all.
        let chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked";
        assert!(check_transfer_coding(chunked, "u").is_ok());
        assert!(check_transfer_coding(b"HTTP/1.1 200 OK\r\nContent-Length: 5", "u").is_ok());
        // §7: coding names are case-insensitive tokens.
        let cased = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: Chunked";
        assert!(check_transfer_coding(cased, "u").is_ok());

        // §6.1: the list is the sequence of codings applied, and we can only undo the
        // outermost one — `parse_framing` still frames this as chunked (§6.3 rule 4),
        // which is why the decodability question has to be asked separately.
        let stacked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip, chunked";
        assert!(check_transfer_coding(stacked, "u").is_err());
        assert!(matches!(
            parse_framing(stacked, "u").unwrap(),
            Framing::Chunked(_)
        ));
        // ... including when the list is spread over field lines (RFC 9110 §5.3).
        let split = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\nTransfer-Encoding: chunked";
        assert!(check_transfer_coding(split, "u").is_err());
        // §6.1: chunked MUST NOT be applied more than once; we decode one layer.
        let twice = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked, chunked";
        assert!(check_transfer_coding(twice, "u").is_err());
    }

    #[test]
    fn content_coding_rejects_only_what_needs_decoding() {
        // RFC 9110 §8.4: a content coding survives de-chunking, so a body under one is
        // not the resource and must not be emitted as if it were.
        assert!(check_content_coding(b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip", "u").is_err());
        let chunked_and_coded =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Encoding: gzip";
        assert!(check_content_coding(chunked_and_coded, "u").is_err());
        // Every coding in the list counts, not just the first (§8.4: they are listed in
        // the order applied), and the names are case-insensitive.
        let listed = b"HTTP/1.1 200 OK\r\nContent-Encoding: identity, BR";
        assert!(check_content_coding(listed, "u").is_err());
        // A folded value must not read as an empty one and slip through (§5.2).
        let folded = b"HTTP/1.1 200 OK\r\nContent-Encoding:\r\n gzip";
        assert!(check_content_coding(folded, "u").is_err());
        // `identity` (§8.4.1) and an absent field are nothing to decode.
        let identity = b"HTTP/1.1 200 OK\r\nContent-Encoding: identity";
        assert!(check_content_coding(identity, "u").is_ok());
        assert!(check_content_coding(b"HTTP/1.1 200 OK\r\nContent-Length: 5", "u").is_ok());
    }

    #[test]
    fn url_rejects_octets_that_would_split_the_request() {
        // CR/LF in the request target ends the request-line and injects field lines of
        // the caller's choosing (RFC 9112 §11.1, §11.2); a bare SP mis-splits it.
        assert!(parse_http_url("http://example.com/a\r\nX-Injected: 1").is_err());
        assert!(parse_http_url("http://example.com\r\nX-Injected: 1/a").is_err());
        assert!(parse_http_url("http://example.com/a\nb").is_err());
        assert!(parse_http_url("http://example.com/a b").is_err());
        assert!(parse_http_url("http://example.com/a\0b").is_err());
        // Percent-encoded, as RFC 3986 §2 requires, it is just a path.
        assert_eq!(
            parse_http_url("http://example.com/a%0d%0ab").unwrap().path,
            "/a%0d%0ab"
        );
    }
}
