//! `httpsrc` — downloads a file over plain HTTP into pooled buffers
//! (spec: Milestone applications §2).
//!
//! This first version does **blocking** networking directly: `start()` opens a
//! [`TcpStream`], sends a `GET`, and parses the status line + headers; `process()`
//! reads the response body in chunks into pooled buffers and pushes them downstream,
//! ending with [`Flow::Eos`] when the server closes the connection. It is an `Active`
//! element (owns its group's thread), so blocking in `process()` is legal and does
//! not stall the rest of the graph.
//!
//! ## v1 limitations
//! - **`http://` only** — no TLS, so no `https`. (Follow-up: a TLS transport.)
//! - **Read-to-EOF** — the body is consumed until the socket closes, which is why the
//!   request sends `Connection: close`. `Content-Length` and `Transfer-Encoding:
//!   chunked` are *not* interpreted; a keep-alive or chunked response would be read
//!   as opaque bytes. (Follow-up: honor framing.)

use std::io::{Read, Write};
use std::net::TcpStream;

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

/// Cap on the response header block, so a server that never sends `\r\n\r\n` can't
/// make us buffer unboundedly.
const MAX_HEADER_BYTES: usize = 64 * 1024;

static PADS: [PadDesc; 1] = [PadDesc {
    name: "src",
    direction: Direction::Src,
    offers: &[],
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

/// The pieces of an `http://` URL this element needs.
struct Url {
    host: String,
    port: u16,
    path: String,
}

/// Parse a plain-`http` URL by hand (core is dependency-free; so are its elements).
/// Extracts host, port (default 80), and path (default `/`). Rejects any non-`http`
/// scheme — TLS/`https` is a documented v1 limitation.
fn parse_http_url(url: &str) -> Result<Url, Error> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| Error::Resource(format!("httpsrc: only http:// URLs supported: {url}")))?;

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
        None => (hostport, 80u16),
    };

    if host.is_empty() {
        return Err(Error::Resource(format!("httpsrc: missing host in {url}")));
    }

    Ok(Url {
        host: host.to_string(),
        port,
        path,
    })
}

/// ```text
/// +--------------------+
/// |               _____|
/// |  HttpSrc     | src |----> bytes
/// |               ^^^^^|
/// +--------------------+
/// ```
pub struct HttpSrc {
    url: String,
    /// Connected + past the response header once `start()` succeeds.
    stream: Option<TcpStream>,
    /// Body bytes that arrived in the same read as the header terminator, not yet
    /// emitted. Drained before reading more from the socket.
    leftover: Vec<u8>,
    /// Set once the socket reports EOF, so `process()` reports `Flow::Eos`.
    eof: bool,
}

impl HttpSrc {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            stream: None,
            leftover: Vec::new(),
            eof: false,
        }
    }
}

/// Read from `stream` until the CRLF-CRLF header terminator, returning any bytes that
/// arrived past it (the start of the body).
fn read_headers(stream: &mut TcpStream, url: &str) -> Result<Vec<u8>, Error> {
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    loop {
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
            return parse_status(&buf, url).map(|()| leftover);
        }

        if buf.len() > MAX_HEADER_BYTES {
            return Err(Error::Resource(format!(
                "httpsrc: response headers exceeded {MAX_HEADER_BYTES} bytes from {url}"
            )));
        }
    }
}

/// Index of the `\r\n\r\n` that separates headers from body, if present.
fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Validate the status line: `HTTP/1.x <code> <reason>`, code must be `2xx`.
fn parse_status(header_block: &[u8], url: &str) -> Result<(), Error> {
    let text = String::from_utf8_lossy(header_block);
    let status_line = text.lines().next().unwrap_or("");
    let mut parts = status_line.split_whitespace();
    let version = parts.next().unwrap_or("");
    if !version.starts_with("HTTP/") {
        return Err(Error::Resource(format!(
            "httpsrc: malformed status line from {url}: {status_line:?}"
        )));
    }
    let code = parts
        .next()
        .and_then(|c| c.parse::<u16>().ok())
        .ok_or_else(|| {
            Error::Resource(format!(
                "httpsrc: malformed status line from {url}: {status_line:?}"
            ))
        })?;
    if !(200..300).contains(&code) {
        return Err(Error::Resource(format!(
            "httpsrc: {url} returned HTTP status {code} ({status_line})"
        )));
    }
    Ok(())
}

impl Element for HttpSrc {
    fn desc(&self) -> &'static ElementDesc {
        &DESC
    }

    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        let url = parse_http_url(&self.url)?;

        let mut stream = TcpStream::connect((url.host.as_str(), url.port))
            .map_err(|e| Error::Resource(format!("httpsrc: connect {}: {e}", self.url)))?;

        let request = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nUser-Agent: streamcraft\r\n\r\n",
            url.path, url.host
        );
        stream
            .write_all(request.as_bytes())
            .map_err(|e| Error::Resource(format!("httpsrc: send request to {}: {e}", self.url)))?;

        // Read + validate the response head; keep any body bytes read past it.
        self.leftover = read_headers(&mut stream, &self.url)?;
        self.stream = Some(stream);
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        // Grab a pooled buffer; `None` is backpressure — try again next call.
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok),
        };

        // First drain leftover header-spill bytes (from the same read as the header
        // terminator), then read more from the socket. Never mix the two in one
        // buffer, to keep the step simple and the ordering obvious.
        if !self.leftover.is_empty() {
            let cap = buf.memory.capacity();
            let take = self.leftover.len().min(cap);
            buf.memory.as_mut_full()[..take].copy_from_slice(&self.leftover[..take]);
            buf.memory.set_len(take);
            self.leftover.drain(..take);
            ctx.out(PadId(0)).push(buf);
            return Ok(Flow::Ok);
        }

        if self.eof {
            return Ok(Flow::Eos);
        }

        let stream = self
            .stream
            .as_mut()
            .ok_or(Error::Todo("httpsrc not started"))?;
        let n = stream
            .read(buf.memory.as_mut_full())
            .map_err(|e| Error::Resource(format!("httpsrc: read body from {}: {e}", self.url)))?;

        if n == 0 {
            // Server closed the connection: end of the download.
            self.eof = true;
            return Ok(Flow::Eos);
        }

        buf.memory.set_len(n);
        ctx.out(PadId(0)).push(buf);
        Ok(Flow::Ok)
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // Drop the connection; the server sees the close.
        self.stream = None;
        self.leftover.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_http_url, parse_status};

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
    fn rejects_https() {
        assert!(parse_http_url("https://example.com/").is_err());
    }

    #[test]
    fn rejects_bad_port() {
        assert!(parse_http_url("http://example.com:notaport/").is_err());
    }

    #[test]
    fn status_2xx_ok_others_err() {
        assert!(parse_status(b"HTTP/1.1 200 OK", "u").is_ok());
        assert!(parse_status(b"HTTP/1.1 204 No Content", "u").is_ok());
        assert!(parse_status(b"HTTP/1.1 404 Not Found", "u").is_err());
        assert!(parse_status(b"HTTP/1.1 500 Internal Server Error", "u").is_err());
        assert!(parse_status(b"garbage", "u").is_err());
    }
}
