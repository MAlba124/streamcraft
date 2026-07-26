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
//! [`OpKind::Recv`](streamcraft_core::io::OpKind::Recv) ops — decodes them into pooled
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
//! `gzip`) is rejected, as this element does not implement content decoders.
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
//! ## v1 limitations
//! - **No client writes after the request**: a server-initiated TLS 1.3 KeyUpdate
//!   or rekey mid-download gets its response queued but never flushed (the reactor
//!   has no streaming send yet); receive-direction decryption is unaffected.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::unix::io::{FromRawFd, IntoRawFd};

use crate::tls;
use streamcraft_core::batch::Inputs;
use streamcraft_core::ctx::Ctx;
use streamcraft_core::element::{
    Direction, Element, ElementDesc, Flow, InputPolicy, LatencyDesc, PadDesc, SchedHint,
};
use streamcraft_core::error::Error;
use streamcraft_core::event::Event;
use streamcraft_core::format::OfferDesc;
use streamcraft_core::id::PadId;
use streamcraft_core::io::{FileHandle, IoResult};
use streamcraft_core::time::Timestamp;

/// Cap on the response header block, so a server that never sends `\r\n\r\n` can't
/// make us buffer unboundedly.
const MAX_HEADER_BYTES: usize = 64 * 1024;

/// Cap on a single chunk-size line (`<hex>[;ext]\r\n`) in a chunked body, so a
/// malformed server that streams an unbounded chunk-size/extension line can't make us
/// buffer unboundedly (RFC 9112 §7.1: recipients MUST anticipate large chunk-size
/// numerals and prevent parsing errors — we bound the line instead).
const MAX_CHUNK_LINE_BYTES: usize = 16 * 1024;

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

/// The pieces of an `http://`/`https://` URL this element needs.
struct Url {
    host: String,
    port: u16,
    path: String,
    /// `https://` — wrap the connection in TLS (see [`crate::tls`]).
    tls: bool,
}

/// Parse an `http`/`https` URL by hand (the HTTP layer stays dependency-free).
/// Extracts host, port (default 80 for `http`, 443 for `https`), and path (default
/// `/`). Any other scheme is rejected.
fn parse_http_url(url: &str) -> Result<Url, Error> {
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
    /// A `Recv` op is outstanding — its bytes have not yet arrived. At most one is in
    /// flight (a socket stream is sequential; see the module docs), so this is a bool,
    /// not a count. `process()` never submits a second read while this is set.
    read_in_flight: bool,
    /// Set once a completed `Recv` returned 0 bytes: the peer closed the socket
    /// (EOF). The decoder treats an exhausted `inbuf` under this flag as end-of-body.
    socket_eof: bool,
    /// Monotonic tag on each submitted `Recv` (rides back as the completion `user`).
    /// Reads submitted below `valid_from` are stale (a flush landed) and discarded —
    /// the same seq-floor `filesrc` uses, needed because `discard_buffers` does not
    /// clear the IO mailbox (an in-flight completion still arrives).
    seq: u64,
    /// Completions carrying a `user` below this are dropped. HTTP is not seekable
    /// without `Range` (not implemented), so a flush cannot reposition the stream;
    /// this only ever bumps to cancel an in-flight read on flush, never to re-seek.
    valid_from: u64,
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

impl HttpSrc {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            file: None,
            inbuf: Vec::new(),
            cursor: 0,
            // Overwritten by `start()`; `Eof` is the conservative default.
            framing: Framing::Eof,
            done: false,
            read_in_flight: false,
            socket_eof: false,
            seq: 0,
            valid_from: 0,
            tls: None,
            tls_closed: false,
        }
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
        if self.read_in_flight || self.socket_eof {
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
        self.read_in_flight = true;
        true
    }

    /// Drain any completed `Recv`, appending its bytes to `inbuf` (or noting EOF).
    /// Returns `true` if fresh input landed, so the caller can try the decoder again.
    /// Stale completions (below the seq floor after a flush) are discarded.
    fn drain_reads(&mut self, ctx: &mut Ctx) -> Result<bool, Error> {
        let mut got = false;
        loop {
            let Some(c) = ctx.io().next_completion() else {
                break;
            };
            self.read_in_flight = false;
            if c.user < self.valid_from {
                continue; // read submitted before a flush: discard (buf recycles)
            }
            match c.result {
                IoResult::Ok(0) => self.socket_eof = true, // peer closed; buf recycles
                IoResult::Ok(_) => {
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
                    return Err(Error::Resource(format!(
                        "httpsrc: read body from {}: {k:?}",
                        self.url
                    )));
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
                // Server closed before the declared length arrived (RFC 9112 §6.3
                // rule 6: the message is incomplete).
                return Err(Error::Resource(format!(
                    "httpsrc: {} closed with {remaining} of Content-Length bytes unread",
                    self.url
                )));
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
                        LineResult::Line(line) => {
                            let size = parse_chunk_size(&line, &self.url)?;
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
                        LineResult::Line(line) => {
                            if line.is_empty() {
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

    /// The decoder needs more bytes than are buffered. If the socket already hit EOF,
    /// the body was truncated mid-framing (a protocol error naming `what`); otherwise
    /// return whatever was decoded so far and ask for another read — emitting the
    /// partial output keeps buffers flowing while the next read is in flight.
    fn need_more_or_short(&self, written: usize, what: &str) -> Result<FillOutcome, Error> {
        if self.socket_eof {
            return Err(Error::Resource(format!(
                "httpsrc: {} closed mid {what}",
                self.url
            )));
        }
        Ok(if written > 0 {
            FillOutcome::Done(written, false)
        } else {
            FillOutcome::NeedMore
        })
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

/// Outcome of a body-fill step (one of `fill_length`/`fill_chunked`/`fill_eof`).
enum FillOutcome {
    /// Wrote `usize` decoded octets into `dst`; `bool` is EOS (body complete).
    Done(usize, bool),
    /// The decoder needs bytes that are not buffered yet and the socket is not at
    /// EOF — `process()` must submit another read and try again on a later pass.
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
/// Generic over the transport: a plain [`TcpStream`], or a [`rustls::Stream`] running
/// the TLS session over it.
// Blocking reads are `start()`-time setup (the response head), not element streaming
// IO — the sanctioned exception to the workspace blocking-IO lint.
#[allow(clippy::disallowed_methods)]
fn read_headers(stream: &mut impl Read, url: &str) -> Result<Head, Error> {
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

    // One-time synchronous setup (connect, TLS handshake, request write, header
    // read) — the sanctioned exception to the workspace blocking-IO lint; the
    // streaming body rides the reactor.
    #[allow(clippy::disallowed_methods)]
    fn start(&mut self, ctx: &mut Ctx) -> Result<(), Error> {
        let url = parse_http_url(&self.url)?;

        // One-time setup stays synchronous (spec: IO — connect + request write are a
        // `start()`-time act, like `filesrc` opening its file): connect, TLS
        // handshake for `https` ([`crate::tls`]), GET, read the response head. Only
        // the streaming *body* reads ride the reactor.
        let mut stream = TcpStream::connect((url.host.as_str(), url.port))
            .map_err(|e| Error::Resource(format!("httpsrc: connect {}: {e}", self.url)))?;

        self.cursor = 0;
        self.done = false;
        self.read_in_flight = false;
        self.socket_eof = false;
        self.seq = 0;
        self.valid_from = 0;
        self.tls_closed = false;
        self.tls = if url.tls {
            Some(tls::connect(&url.host, &mut stream, &self.url)?)
        } else {
            None
        };

        // `Host` carries the port when it isn't the scheme default (RFC 9112 §3.2:
        // Host is the target URI's authority) — virtual hosts on nonstandard ports
        // route on it.
        let default_port = if url.tls { 443 } else { 80 };
        let host_header = if url.port == default_port {
            url.host.clone()
        } else {
            format!("{}:{}", url.host, url.port)
        };
        let request = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nUser-Agent: streamcraft\r\n\r\n",
            url.path, host_header
        );

        // Write the request, then read + validate the response head, pick the body
        // framing (RFC 9112 §6.3), and seed the decoder's input with any body bytes
        // that arrived in the same read as the header terminator — through the TLS
        // session when there is one ([`rustls::Stream`] runs the state machine over
        // the socket synchronously), else straight over TCP.
        let head = match self.tls.as_mut() {
            Some(conn) => {
                let mut s = rustls::Stream::new(conn, &mut stream);
                s.write_all(request.as_bytes()).map_err(|e| {
                    Error::Resource(format!("httpsrc: send request to {}: {e}", self.url))
                })?;
                let mut head = read_headers(&mut s, &self.url)?;
                // `read_headers` stopped at the header terminator, but the TLS record
                // that carried it may have decrypted *more* plaintext than the header
                // read consumed — still buffered in the session. Move it into the
                // leftover now: `process()` only feeds the session ciphertext from
                // new completions and would never see these bytes.
                let state = s.conn.process_new_packets().map_err(|e| {
                    Error::Resource(format!("httpsrc: TLS error from {}: {e}", self.url))
                })?;
                let plain = state.plaintext_bytes_to_read();
                if plain > 0 {
                    let start = head.leftover.len();
                    head.leftover.resize(start + plain, 0);
                    // In-memory drain of already-decrypted bytes — no fd, no
                    // blocking (see `decrypt_into_inbuf`).
                    s.conn.reader().read_exact(&mut head.leftover[start..]).map_err(
                        |e| Error::Resource(format!("httpsrc: TLS read from {}: {e}", self.url)),
                    )?;
                }
                if state.peer_has_closed() {
                    self.tls_closed = true;
                    self.socket_eof = true;
                }
                head
            }
            None => {
                stream.write_all(request.as_bytes()).map_err(|e| {
                    Error::Resource(format!("httpsrc: send request to {}: {e}", self.url))
                })?;
                read_headers(&mut stream, &self.url)?
            }
        };
        self.framing = head.framing;
        self.inbuf = head.leftover;

        // Hand the connected socket to the reactor. `into_raw_fd` consumes the
        // `TcpStream` (releasing sole fd ownership without closing it); `File`
        // adopts that fd, and the reactor owns/closes it thereafter — the same
        // registration path `filesrc` uses, so body reads become submitted ops.
        // SAFETY: the fd came from `into_raw_fd`, which transferred ownership to us.
        let file = unsafe { std::fs::File::from_raw_fd(stream.into_raw_fd()) };
        self.file = Some(ctx.io().register(file));
        Ok(())
    }

    fn process(&mut self, ctx: &mut Ctx, _inputs: Inputs<'_>) -> Result<Flow, Error> {
        if self.done {
            return Ok(Flow::Eos);
        }

        // Drain any completed body read into `inbuf` (frees the read-in-flight slot).
        self.drain_reads(ctx)?;

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
            buf.memory.set_len(written);
            ctx.out(PadId(0)).push(buf);
        }

        Ok(if self.done { Flow::Eos } else { Flow::Ok })
    }

    fn event(&mut self, _ctx: &mut Ctx, event: &Event) -> Result<(), Error> {
        // HTTP is not seekable without a `Range` request (not implemented), so a flush
        // cannot reposition the stream — we intentionally ignore the *target*. But a
        // flush must not leak an in-flight read: `discard_buffers` does not clear the
        // IO mailbox, so its completion still arrives. Bump the seq floor so that
        // completion is discarded on arrival (the same guard `filesrc` uses).
        if matches!(event, Event::FlushStart) {
            self.valid_from = self.seq;
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &mut Ctx) {
        // The reactor owns and closes the socket fd (registered in `start`). Just drop
        // our handle and buffered state. Dropping the TLS session without close_notify
        // is fine: we are the reader, with nothing left to authenticate to the peer.
        self.file = None;
        self.inbuf.clear();
        self.cursor = 0;
        self.read_in_flight = false;
        self.socket_eof = false;
        self.tls = None;
        self.tls_closed = false;
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
