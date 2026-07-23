//! `httpsrc` — downloads a file over plain HTTP into pooled buffers
//! (spec: Milestone applications §2).
//!
//! This version does **blocking** networking directly: `start()` opens a
//! [`TcpStream`], sends a `GET`, parses the status line + headers, and picks a
//! **message-body framing mode** from those headers per RFC 9112 (checked in at
//! `spec/rfc9112.txt`). `process()` decodes the response body into pooled buffers and
//! pushes them downstream, ending with [`Flow::Eos`] once the body is complete. It is
//! an `Active` element (owns its group's thread), so blocking in `process()` is legal
//! and does not stall the rest of the graph.
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
//! `gzip`) is rejected, as this element does not implement content decoders.
//!
//! ## v1 limitations
//! - **`http://` only** — no TLS, so no `https`. (Follow-up: a TLS transport.)

use std::io::{Read, Write};
use std::net::TcpStream;

use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::time::Timestamp;

/// Cap on the response header block, so a server that never sends `\r\n\r\n` can't
/// make us buffer unboundedly.
const MAX_HEADER_BYTES: usize = 64 * 1024;

/// Cap on a single chunk-size line (`<hex>[;ext]\r\n`) in a chunked body, so a
/// malformed server that streams an unbounded chunk-size/extension line can't make us
/// buffer unboundedly (RFC 9112 §7.1: recipients MUST anticipate large chunk-size
/// numerals and prevent parsing errors — we bound the line instead).
const MAX_CHUNK_LINE_BYTES: usize = 16 * 1024;

/// How much we pull from the socket per fill when the decoder needs more input.
const READ_CHUNK: usize = 64 * 1024;

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
    url: String,
    /// Connected + past the response header once `start()` succeeds.
    stream: Option<TcpStream>,
    /// Undecoded body bytes already pulled from the socket but not yet consumed by the
    /// framing decoder. Seeded in `start()` with the bytes that arrived in the same
    /// read as the header terminator (a Content-Length/chunked body often begins in
    /// that read), then refilled from the socket as the decoder needs more.
    inbuf: Vec<u8>,
    /// Read cursor into `inbuf`: bytes `[..cursor]` are consumed. We advance the
    /// cursor instead of draining so the decoder can work in slices without shifting
    /// the buffer on every step; it is compacted when refilling.
    cursor: usize,
    /// How the body is framed, chosen from the response headers in `start()`.
    framing: Framing,
    /// Set once the body is fully delivered, so `process()` reports `Flow::Eos`.
    done: bool,
}

impl HttpSrc {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            stream: None,
            inbuf: Vec::new(),
            cursor: 0,
            // Overwritten by `start()`; `Eof` is the conservative default.
            framing: Framing::Eof,
            done: false,
        }
    }

    /// Undecoded body bytes not yet consumed by the decoder.
    fn avail(&self) -> &[u8] {
        &self.inbuf[self.cursor..]
    }

    /// Pull more undecoded body bytes from the socket into `inbuf`. First compacts by
    /// dropping the already-consumed prefix `[..cursor]` (so the buffer can't grow
    /// without bound across chunks), then reads one block. Returns the number of bytes
    /// read; `0` means the socket reached EOF. Blocking is fine — `HttpSrc` is
    /// `Active`.
    fn fill_socket(&mut self) -> Result<usize, Error> {
        if self.cursor > 0 {
            self.inbuf.drain(..self.cursor);
            self.cursor = 0;
        }
        let stream = self
            .stream
            .as_mut()
            .ok_or(Error::Todo("httpsrc not started"))?;
        let base = self.inbuf.len();
        self.inbuf.resize(base + READ_CHUNK, 0);
        let n = stream
            .read(&mut self.inbuf[base..])
            .map_err(|e| Error::Resource(format!("httpsrc: read body from {}: {e}", self.url)))?;
        self.inbuf.truncate(base + n);
        Ok(n)
    }

    /// `Content-Length` body (RFC 9112 §6.2): emit up to `dst.len()` of the remaining
    /// octets, then signal EOS once the declared count is exhausted — without waiting
    /// for the socket to close. Buffered leftover is emitted first; otherwise we read
    /// straight into `dst` to avoid double-buffering a large body.
    fn fill_length(&mut self, dst: &mut [u8]) -> Result<(usize, bool), Error> {
        let Framing::Length(remaining) = self.framing else {
            unreachable!("fill_length called for non-length framing");
        };
        if remaining == 0 {
            return Ok((0, true));
        }
        // How many octets we may take this step: bounded by both the output buffer and
        // the declared remaining length. `usize::try_from` guards a >usize length on
        // 32-bit targets (we just cap at the buffer size, which is always fine).
        let want = dst
            .len()
            .min(usize::try_from(remaining).unwrap_or(usize::MAX));

        // Prefer buffered leftover bytes (from the header read).
        let buffered = self.avail().len();
        let n = if buffered > 0 {
            let take = buffered.min(want);
            dst[..take].copy_from_slice(&self.avail()[..take]);
            self.cursor += take;
            take
        } else {
            let stream = self
                .stream
                .as_mut()
                .ok_or(Error::Todo("httpsrc not started"))?;
            stream.read(&mut dst[..want]).map_err(|e| {
                Error::Resource(format!("httpsrc: read body from {}: {e}", self.url))
            })?
        };

        if n == 0 {
            // Server closed before the declared length arrived (RFC 9112 §6.3 rule 6:
            // the message is incomplete).
            return Err(Error::Resource(format!(
                "httpsrc: {} closed with {remaining} of Content-Length bytes unread",
                self.url
            )));
        }

        let left = remaining - n as u64;
        self.framing = Framing::Length(left);
        Ok((n, left == 0))
    }

    /// Close-delimited body (RFC 9112 §6.3 rule 8): emit buffered leftover first, then
    /// read the socket until it reports EOF, which ends the download.
    fn fill_eof(&mut self, dst: &mut [u8]) -> Result<(usize, bool), Error> {
        let buffered = self.avail().len();
        if buffered > 0 {
            let take = buffered.min(dst.len());
            dst[..take].copy_from_slice(&self.avail()[..take]);
            self.cursor += take;
            return Ok((take, false));
        }
        let stream = self
            .stream
            .as_mut()
            .ok_or(Error::Todo("httpsrc not started"))?;
        let n = stream
            .read(dst)
            .map_err(|e| Error::Resource(format!("httpsrc: read body from {}: {e}", self.url)))?;
        Ok((n, n == 0))
    }

    /// `chunked` transfer coding (RFC 9112 §7.1). Runs the chunk state machine over
    /// buffered input, refilling from the socket as needed, and writes as many decoded
    /// `chunk-data` octets into `dst` as fit. Returns `(written, eos)`; `eos` is set
    /// once the zero last-chunk and its terminating CRLF have been consumed.
    ///
    /// The step returns as soon as `dst` is full *or* the decoder needs bytes that a
    /// socket read did not yet provide, so a single `process()` call never spins.
    fn fill_chunked(&mut self, dst: &mut [u8]) -> Result<(usize, bool), Error> {
        let mut written = 0;
        loop {
            let state = match self.framing {
                Framing::Chunked(s) => s,
                _ => unreachable!("fill_chunked called for non-chunked framing"),
            };

            match state {
                ChunkState::Done => return Ok((written, true)),

                ChunkState::Size => {
                    // Need a full `chunk-size [ chunk-ext ] CRLF` line before decoding.
                    match self.take_line(MAX_CHUNK_LINE_BYTES)? {
                        LineResult::Line(line) => {
                            let size = parse_chunk_size(&line, &self.url)?;
                            self.framing = Framing::Chunked(if size == 0 {
                                ChunkState::Trailer // zero = last-chunk (§7.1)
                            } else {
                                ChunkState::Data(size)
                            });
                        }
                        LineResult::NeedMore => {
                            if self.fill_socket()? == 0 {
                                return Err(Error::Resource(format!(
                                    "httpsrc: {} closed mid chunk-size line",
                                    self.url
                                )));
                            }
                        }
                    }
                }

                ChunkState::Data(remaining) => {
                    // Copy as much chunk-data as fits in `dst` and is buffered.
                    let space = dst.len() - written;
                    if space == 0 {
                        return Ok((written, false)); // buffer full; resume next step
                    }
                    let buffered = self.avail().len();
                    if buffered == 0 {
                        if self.fill_socket()? == 0 {
                            return Err(Error::Resource(format!(
                                "httpsrc: {} closed mid chunk-data",
                                self.url
                            )));
                        }
                        continue;
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
                        None => {
                            if self.fill_socket()? == 0 {
                                return Err(Error::Resource(format!(
                                    "httpsrc: {} closed before chunk CRLF",
                                    self.url
                                )));
                            }
                        }
                    }
                }

                ChunkState::Trailer => {
                    // After the zero last-chunk: consume `trailer-section CRLF`, i.e.
                    // zero or more `field-line CRLF` then the terminating empty line
                    // (§7.1.2). We ignore trailer field contents entirely.
                    match self.take_line(MAX_CHUNK_LINE_BYTES)? {
                        LineResult::Line(line) => {
                            if line.is_empty() {
                                // The empty line terminates the body.
                                self.framing = Framing::Chunked(ChunkState::Done);
                                return Ok((written, true));
                            }
                            // A trailer field line — skip it and keep reading.
                        }
                        LineResult::NeedMore => {
                            if self.fill_socket()? == 0 {
                                return Err(Error::Resource(format!(
                                    "httpsrc: {} closed in chunked trailer",
                                    self.url
                                )));
                            }
                        }
                    }
                }
            }

            // If we filled the output buffer, hand it off; otherwise loop to make more
            // progress (the socket reads above are what bound the loop).
            if written == dst.len() {
                return Ok((written, false));
            }
        }
    }

    /// Try to take one CRLF-terminated line from buffered input, *without* the CRLF.
    /// Returns [`LineResult::NeedMore`] if no complete line is buffered yet. Bounds the
    /// scanned length by `max` (RFC 9112 §7.1: guard against unbounded chunk-size /
    /// extension lines) — a longer line is a protocol error.
    fn take_line(&mut self, max: usize) -> Result<LineResult, Error> {
        let buf = self.avail();
        if let Some(i) = find_crlf(buf) {
            let line = buf[..i].to_vec();
            self.cursor += i + 2; // consume the line and its CRLF
            Ok(LineResult::Line(line))
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
}

/// Outcome of trying to read a full line out of buffered input.
enum LineResult {
    /// A complete line (CRLF stripped).
    Line(Vec<u8>),
    /// No complete line buffered yet — read more from the socket and retry.
    NeedMore,
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

/// The response head, once parsed: the chosen body framing plus any body bytes that
/// arrived in the same socket read as the header terminator.
struct Head {
    /// How the message body is framed (RFC 9112 §6.3).
    framing: Framing,
    /// Bytes read past `\r\n\r\n` — the start of the body, to feed the decoder.
    leftover: Vec<u8>,
}

/// Read from `stream` until the CRLF-CRLF header terminator, validate the status line,
/// and pick the body framing (RFC 9112 §6.3) from the header fields. Returns the
/// framing plus any bytes that arrived past the terminator (the start of the body).
fn read_headers(stream: &mut TcpStream, url: &str) -> Result<Head, Error> {
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
            parse_status(&buf, url)?;
            let framing = parse_framing(&buf, url)?;
            return Ok(Head { framing, leftover });
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

/// Return the last value of header `name` (case-insensitive, RFC 9112 header names are
/// case-insensitive tokens), trimmed of surrounding whitespace. `header_block` is the
/// status line plus header lines, without the trailing CRLF-CRLF. Later occurrences
/// win, which matters for none of our fields but keeps the lookup unsurprising.
fn header_value<'a>(header_block: &'a str, name: &str) -> Option<&'a str> {
    header_block
        .lines()
        .skip(1) // the status line, not a header field
        .filter_map(|line| line.split_once(':'))
        .filter(|(k, _)| k.trim().eq_ignore_ascii_case(name))
        .map(|(_, v)| v.trim())
        .last()
}

/// Choose the message-body framing from the response header block (RFC 9112 §6.3).
///
/// Precedence: `Transfer-Encoding` overrides `Content-Length` (§6.3 rule 3); a valid
/// `Content-Length` frames the body otherwise (§6.2); with neither, the body is read
/// to EOF (§6.3 rule 8). Only the `chunked` transfer coding — and only as the final
/// coding — is decoded; any other coding is rejected, since we implement no content
/// decoders.
fn parse_framing(header_block: &[u8], url: &str) -> Result<Framing, Error> {
    let text = String::from_utf8_lossy(header_block);

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

    // RFC 9112 §6.2: a decimal octet count. §6.3 rule 5 allows a comma-separated list
    // only if every value is identical; we accept a single value and otherwise treat
    // the framing as invalid.
    if let Some(cl) = header_value(&text, "Content-Length") {
        let mut values = cl.split(',').map(str::trim);
        let first = values.next().unwrap_or("");
        let len: u64 = first.parse().map_err(|_| {
            Error::Resource(format!(
                "httpsrc: invalid Content-Length {cl:?} from {url}"
            ))
        })?;
        if !values.all(|v| v == first) {
            return Err(Error::Resource(format!(
                "httpsrc: conflicting Content-Length {cl:?} from {url}"
            )));
        }
        return Ok(Framing::Length(len));
    }

    // RFC 9112 §6.3 rule 8: no declared length → close-delimited body.
    Ok(Framing::Eof)
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

        // Read + validate the response head, pick the body framing (RFC 9112 §6.3),
        // and seed the decoder's input with any body bytes that arrived in the same
        // read as the header terminator.
        let head = read_headers(&mut stream, &self.url)?;
        self.framing = head.framing;
        self.inbuf = head.leftover;
        self.cursor = 0;
        self.done = false;
        self.stream = Some(stream);
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.done {
            return Ok(Flow::Eos);
        }

        // Grab a pooled buffer; `None` is backpressure — try again next call.
        let mut buf = match ctx.try_alloc(PadId(0)) {
            Some(b) => b,
            None => return Ok(Flow::Ok),
        };

        // Fill the buffer with decoded body bytes according to the framing. `written`
        // is how many landed in it; `eos` means the body is complete this step.
        let cap = buf.memory.capacity();
        let dst = buf.memory.as_mut_full();
        let (written, eos) = match self.framing {
            Framing::Length(_) => self.fill_length(&mut dst[..cap])?,
            Framing::Chunked(_) => self.fill_chunked(&mut dst[..cap])?,
            Framing::Eof => self.fill_eof(&mut dst[..cap])?,
        };

        if eos {
            self.done = true;
        }

        // Push non-empty buffers; skip empty ones so we don't emit a zero-length
        // buffer just to carry EOS (e.g. an empty chunked step that only consumed
        // framing bytes). `try_alloc` again next call is cheap.
        if written > 0 {
            buf.memory.set_len(written);
            ctx.out(PadId(0)).push(buf);
        }

        Ok(if self.done { Flow::Eos } else { Flow::Ok })
    }

    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // Drop the connection; the server sees the close.
        self.stream = None;
        self.inbuf.clear();
        self.cursor = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        header_value, parse_chunk_size, parse_framing, parse_http_url, parse_status, Framing,
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
}
