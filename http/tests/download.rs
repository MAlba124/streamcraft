//! Milestone 2: `httpsrc ! filesink` downloads a file over HTTP and writes it to
//! disk byte-identically (spec: Milestone applications §2).
//!
//! Hermetic — no real internet (the one real-world spot check is `#[ignore]`d). Each
//! test stands up a `std::net::TcpListener` on `127.0.0.1:0`, serves a canned or
//! scripted response from a background thread, and points the pipeline at it.
//!
//! Later tests grow that into what a podcast player needs: redirect chains, byte
//! ranges, resume after a dropped connection, stall recovery and seeking.

// The test *server* is app-side code, not element code: it writes to its sockets with
// blocking `write_all` and builds its fixtures on the heap, which is exactly the
// exception `clippy.toml` names ("app `main()` setup, tests"). The element under test
// is held to the real rule — `http/src/httpsrc.rs` is clippy-clean without this.
#![allow(clippy::disallowed_methods)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pf_http::HttpSrc;
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
use profluens_core::pipeline::Pipeline;
use profluens_core::time::Timestamp;
use profluens_elements::io::FileSink;

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf_m2_{}_{}.bin", tag, std::process::id()));
    p
}

/// Read a full HTTP request head (up to and including the blank line) off a stream,
/// returning it as text so a test can assert on the request line and its fields.
fn read_request(stream: &mut std::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 512];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => break,
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

/// Spawn a one-shot HTTP server: accept a single connection, read the request, send
/// `head` immediately followed by `body`, then close. Returns the bound port.
fn serve_once(head: &'static str, body: Vec<u8>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        read_request(&mut stream);
        // Write the head, then the body. `write_all` blocks until the client drains,
        // which is exactly the backpressure path we want to exercise for a large body.
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(&body);
        // Dropping `stream` (and the listener) closes the connection → client sees EOF.
    });
    port
}

/// Like [`serve_once`], but the server does **not** close after writing: it holds the
/// connection open until the returned [`Sender`] is dropped (or signalled). This proves
/// a length- or chunk-framed download terminates on its own framing rather than on the
/// socket closing — if the client wrongly waited for EOF, the test would hang.
///
/// The test must keep the returned guard alive until *after* `run()` returns, then drop
/// it to let the server thread close.
fn serve_once_no_close(head: &'static str, body: Vec<u8>) -> (u16, std::sync::mpsc::Sender<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        read_request(&mut stream);
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(&body);
        // Do NOT drop `stream` yet — keep the connection open so the client cannot be
        // relying on EOF to terminate. Block until the test signals it is done.
        let _ = rx.recv();
        // Now the connection (and listener) drop, closing the socket.
    });
    (port, tx)
}

/// Serve `head` then dribble `body` out in `piece`-sized writes, sleeping `gap`
/// between them, and close at the end. This forces the client's reactor read path to
/// span many `process()` passes: each write lands as its own `Recv` completion, so the
/// framing decoder repeatedly starves (`NeedMore`) and resumes as bytes trickle in —
/// the streaming case a single blocking read would have hidden. Returns the port.
fn serve_slow(head: &'static str, body: Vec<u8>, piece: usize, gap: Duration) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        read_request(&mut stream);
        let _ = stream.write_all(head.as_bytes());
        for chunk in body.chunks(piece.max(1)) {
            if stream.write_all(chunk).is_err() {
                return;
            }
            std::thread::sleep(gap);
        }
        // Drop closes the connection → client sees EOF (for the close-delimited case).
    });
    port
}

/// Serve `head` then only `sent` of `total` promised bytes and close — an early EOF
/// under a `Content-Length: total`. The client must error (RFC 9112 §6.3 rule 6:
/// truncated message), not silently accept a short body. Returns the port.
fn serve_early_eof(head: &'static str, sent: Vec<u8>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        read_request(&mut stream);
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(&sent);
        // Close early — fewer than the promised Content-Length bytes were sent.
    });
    port
}

/// Encode `body` as an HTTP/1.1 `chunked` message body (RFC 9112 §7.1): a series of
/// `<hex-size>\r\n<data>\r\n` chunks split at `sizes`, then the `0\r\n\r\n` terminator.
fn chunked_encode(body: &[u8], sizes: &[usize]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut off = 0;
    for &n in sizes {
        let end = (off + n).min(body.len());
        let piece = &body[off..end];
        out.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
        out.extend_from_slice(piece);
        out.extend_from_slice(b"\r\n");
        off = end;
    }
    // Any remaining bytes go in a final chunk, so `sizes` need not cover the whole body.
    if off < body.len() {
        let piece = &body[off..];
        out.extend_from_slice(format!("{:x}\r\n", piece.len()).as_bytes());
        out.extend_from_slice(piece);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(b"0\r\n\r\n"); // last-chunk + terminating CRLF
    out
}

#[test]
fn downloads_file_byte_identically() {
    // ~500 KiB of non-trivial content: larger than the 128 KiB pool slot, so several
    // buffers/batches flow, and larger than a socket buffer, so the server blocks on
    // write until the pipeline reads.
    let body: Vec<u8> = (0..512 * 1024u32).map(|i| (i % 251) as u8).collect();
    let port = serve_once("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n", body.clone());

    let outp = temp_path("ok_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(
        got.len(),
        body.len(),
        "downloaded length matches served body"
    );
    assert_eq!(got, body, "downloaded bytes match served body");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn body_that_spills_into_header_read_is_preserved() {
    // Small body that the server sends in the *same* write as the headers, so it
    // arrives in the client's header read and must be replayed from `leftover`.
    let body = b"hello, profluens body".to_vec();
    let port = serve_once("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n", body.clone());

    let outp = temp_path("spill_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/small")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got, body, "small body read alongside headers is preserved");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn non_2xx_status_errors() {
    // A 404 response must fail the run (the error rides the bus and is returned).
    let port = serve_once(
        "HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n",
        b"nope".to_vec(),
    );

    let outp = temp_path("404_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/missing")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");

    assert!(p.run().is_err(), "404 response must make run() error");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn connection_refused_errors() {
    // Bind then immediately drop the listener to obtain a port nothing is listening
    // on; connecting there should be refused and make the run error.
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    drop(listener);

    let outp = temp_path("refused_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");

    assert!(p.run().is_err(), "connecting to a closed port must error");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn content_length_terminates_without_close() {
    // A `Content-Length` response (RFC 9112 §6.2). The server sends exactly N bytes and
    // then keeps the connection open — the client must stop after N bytes on the length
    // alone, not wait for EOF. ~300 KiB so several pooled buffers flow.
    let body: Vec<u8> = (0..300 * 1024u32).map(|i| (i.wrapping_mul(31) % 253) as u8).collect();
    let head: &'static str = Box::leak(
        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_boxed_str(),
    );
    let (port, done) = serve_once_no_close(head, body.clone());

    let outp = temp_path("clen_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");
    drop(done); // let the server thread close now that the download finished

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got.len(), body.len(), "Content-Length body length matches");
    assert_eq!(got, body, "Content-Length body is byte-identical");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn chunked_transfer_encoding_is_decoded() {
    // A `Transfer-Encoding: chunked` response (RFC 9112 §6.1, §7.1) with chunks of
    // varied sizes plus the zero terminator. The decoded stream must equal the original
    // body. The server also holds the connection open, so termination comes from the
    // last-chunk, not from EOF.
    let body: Vec<u8> = (0..250 * 1024u32).map(|i| (i % 251) as u8).collect();
    // Deliberately irregular chunk sizes, including one crossing the 128 KiB pool slot;
    // `chunked_encode` puts the tail (~35 KiB) in a final chunk.
    let sizes = [1usize, 5, 200, 4096, 130 * 1024, 7, 63 * 1024, 0xFED];
    let encoded = chunked_encode(&body, &sizes);
    let (port, done) = serve_once_no_close(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
        encoded,
    );

    let outp = temp_path("chunked_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");
    drop(done);

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got.len(), body.len(), "decoded chunked length matches body");
    assert_eq!(got, body, "decoded chunked bytes are byte-identical");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn chunked_with_extensions_and_trailer_is_decoded() {
    // Exercise chunk-extensions on the size line and a trailer section after the zero
    // chunk (RFC 9112 §7.1.1, §7.1.2) — both must be ignored, data decoded intact.
    let mut encoded = Vec::new();
    encoded.extend_from_slice(b"5;ext=1\r\nhello\r\n"); // chunk-ext after size
    encoded.extend_from_slice(b"6;a;b=c\r\n world\r\n"); // multiple exts
    encoded.extend_from_slice(b"0\r\n"); // last-chunk
    encoded.extend_from_slice(b"X-Trailer: value\r\n"); // a trailer field (ignored)
    encoded.extend_from_slice(b"\r\n"); // terminating CRLF

    let (port, done) = serve_once_no_close(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
        encoded,
    );

    let outp = temp_path("chunked_ext_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");
    drop(done);

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got, b"hello world", "chunk-ext/trailer ignored, data intact");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn slow_chunked_delivery_reassembles_across_reads() {
    // The reactor read path proven under **trickle** delivery: the chunked body is
    // written 137 bytes at a time with a gap, so most `Recv` completions land a
    // fraction of a chunk. The decoder must span passes (starve → read → resume)
    // without corrupting the byte stream or spinning on partial framing.
    let body: Vec<u8> = (0..64 * 1024u32).map(|i| (i.wrapping_mul(97) % 251) as u8).collect();
    let sizes = [3usize, 1, 5000, 17, 20000, 1];
    let encoded = chunked_encode(&body, &sizes);
    let port = serve_slow(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
        encoded,
        137,
        Duration::from_micros(200),
    );

    let outp = temp_path("slow_chunked_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got.len(), body.len(), "slow chunked length matches");
    assert_eq!(got, body, "slow chunked bytes are byte-identical");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn slow_length_delivery_reassembles_across_reads() {
    // Same trickle stress on the `Content-Length` path: the client must accumulate
    // exactly N bytes across many small reads and terminate on the length.
    let body: Vec<u8> = (0..48 * 1024u32).map(|i| (i.wrapping_mul(53) % 249) as u8).collect();
    let head: &'static str = Box::leak(
        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_boxed_str(),
    );
    let port = serve_slow(head, body.clone(), 211, Duration::from_micros(150));

    let outp = temp_path("slow_len_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");

    let got = std::fs::read(&outp).expect("read output");
    assert_eq!(got.len(), body.len(), "slow Content-Length length matches");
    assert_eq!(got, body, "slow Content-Length bytes are byte-identical");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn content_length_early_eof_errors() {
    // The server promises 1000 bytes but sends 400 then closes. RFC 9112 §6.3 rule 6:
    // the message is incomplete → the run must error, not accept a truncated body.
    let head = "HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n";
    let sent: Vec<u8> = (0..400u32).map(|i| i as u8).collect();
    let port = serve_early_eof(head, sent);

    let outp = temp_path("early_eof_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");

    assert!(p.run().is_err(), "a truncated Content-Length body must error");

    let _ = std::fs::remove_file(&outp);
}

#[test]
fn empty_eof_body_terminates() {
    // A close-delimited response with a zero-length body: the server closes right
    // after the headers. The client must terminate cleanly (EOS on the first empty
    // read), producing an empty output rather than hanging.
    let port = serve_once("HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n", Vec::new());

    let outp = temp_path("empty_eof_out");
    let mut p = Pipeline::new();
    let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/empty")));
    let sink = p.add(FileSink::new(&outp));
    p.link((src, "src"), (sink, "sink")).expect("link");
    p.run().expect("run");

    let got = std::fs::read(&outp).expect("read output");
    assert!(got.is_empty(), "empty close-delimited body yields empty output");

    let _ = std::fs::remove_file(&outp);
}

/// The streaming `Recv` op on a **real** io_uring ring: the reactor issues
/// `IORING_OP_READ` with offset -1 against the socket fd, so this proves the async
/// path end-to-end (not just the SyncReactor's blocking `read(2)`). Both framings run
/// through it; the download must still be byte-identical.
#[cfg(feature = "io-uring")]
#[test]
fn downloads_over_io_uring_reactor() {
    use profluens_elements::io::IoUringReactor;

    fn run_over_uring(head: &'static str, wire: Vec<u8>, tag: &str) -> Vec<u8> {
        let port = serve_once(head, wire);
        let outp = temp_path(tag);
        let mut p = Pipeline::new();
        p.set_reactor_factory(std::sync::Arc::new(|| {
            Ok(Box::new(IoUringReactor::new()?) as Box<dyn profluens_core::io::Reactor>)
        }));
        let src = p.add(HttpSrc::new(format!("http://127.0.0.1:{port}/file")));
        let sink = p.add(FileSink::new(&outp));
        p.link((src, "src"), (sink, "sink")).expect("link");
        p.run().expect("run");
        let got = std::fs::read(&outp).expect("read output");
        let _ = std::fs::remove_file(&outp);
        got
    }

    // Close-delimited (EOF framing): several pool slots' worth of body.
    let body: Vec<u8> = (0..400 * 1024u32).map(|i| (i % 251) as u8).collect();
    let got = run_over_uring(
        "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n",
        body.clone(),
        "uring_eof_out",
    );
    assert_eq!(got, body, "io_uring EOF-framed download is byte-identical");

    // Chunked framing over the same ring.
    let sizes = [1usize, 5000, 130 * 1024, 7, 63 * 1024];
    let encoded = chunked_encode(&body, &sizes);
    let got = run_over_uring(
        "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n",
        encoded,
        "uring_chunked_out",
    );
    assert_eq!(got, body, "io_uring chunked download is byte-identical");
}

// =====================================================================================
// Redirects, ranges, resume and stall recovery — what a podcast player needs on top of
// "download a file". The harness below is the same hermetic local-server style as
// above, grown a script: one thread accepts connections in a loop, hands each to a
// closure with its request head, and logs every head so a test can assert on the
// `Range`/`If-Range`/`Host` fields the client actually sent.
// =====================================================================================

/// A scripted local HTTP server. `handler` is called once per accepted connection with
/// the connection's index (so a script can behave differently the second time) and the
/// request head it read.
struct Server {
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

impl Server {
    /// `http://127.0.0.1:<port><path>`.
    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    /// Every request head received so far, in order.
    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

fn serve_script<F>(handler: F) -> Server
where
    F: Fn(usize, &str, &mut TcpStream) + Send + Sync + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&requests);
    let handler = Arc::new(handler);
    std::thread::spawn(move || {
        for (n, stream) in listener.incoming().enumerate() {
            let Ok(mut stream) = stream else { break };
            // One thread per connection: a script that holds a connection open (the
            // stall test sits silent on it for a minute) must not stop the listener
            // from accepting the reconnect that the stall is supposed to provoke.
            let log = Arc::clone(&log);
            let handler = Arc::clone(&handler);
            std::thread::spawn(move || {
                let head = read_request(&mut stream);
                log.lock().unwrap().push(head.clone());
                handler(n, &head, &mut stream);
            });
        }
    });
    Server { port, requests }
}

/// The request target off a request head (`GET <target> HTTP/1.1`).
fn request_target(head: &str) -> &str {
    head.lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .unwrap_or("")
}

/// A body with enough structure that a wrong offset shows up as a mismatch rather than
/// a coincidence.
fn ramp(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i.wrapping_mul(31) % 251) as u8).collect()
}

/// Write a head + body, ignoring write errors — a client that seeks or reconnects
/// hangs up mid-write on purpose, and the script must not panic on the far end.
fn write_all(stream: &mut TcpStream, head: &str, body: &[u8]) {
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

/// Dribble a body out in `piece`-sized writes so the download spans many `process()`
/// passes and a test thread has time to seek into the middle of it.
fn write_slow(stream: &mut TcpStream, head: &str, body: &[u8], piece: usize, gap: Duration) {
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    for chunk in body.chunks(piece.max(1)) {
        if stream.write_all(chunk).is_err() {
            return;
        }
        std::thread::sleep(gap);
    }
}

/// Poll `cond` until it comes true or `limit` elapses. Returns whether it came true.
fn settle_within(limit: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let began = Instant::now();
    while began.elapsed() < limit {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    cond()
}

/// Poll `cond` for up to ~2 s. Returns whether it came true.
fn settle(cond: impl FnMut() -> bool) -> bool {
    settle_within(Duration::from_secs(2), cond)
}

// --- a sink that records every buffer's bytes and DISCONT flag ------------------------

static REC_OFFERS: [OfferDesc; 1] = [OfferDesc::any("bytes")];
static REC_PADS: [PadDesc; 1] = [PadDesc {
    name: "sink",
    direction: Direction::Sink,
    offers: &REC_OFFERS,
    dynamic: false,
    validate: None,
}];
static REC_DESC: ElementDesc = ElementDesc {
    name: "recsink",
    pads: &REC_PADS,
    props: &[],
    sched: SchedHint::Active,
    inputs: InputPolicy::Single,
    latency: LatencyDesc {
        min: Timestamp::ZERO,
        max: Timestamp::ZERO,
        is_live: false,
        jitter: Timestamp::ZERO,
    },
    make_default: None,
};

#[derive(Default, Debug)]
struct Recorded {
    /// One entry per received buffer: (bytes, carried DISCONT).
    bufs: Vec<(Vec<u8>, bool)>,
}

impl Recorded {
    /// Every byte recorded, in arrival order.
    fn all(&self) -> Vec<u8> {
        self.bufs.iter().flat_map(|(b, _)| b.iter().copied()).collect()
    }

    /// Bytes from the first DISCONT-marked buffer on — the post-seek stream.
    fn after_discont(&self) -> Option<Vec<u8>> {
        let at = self.bufs.iter().position(|(_, d)| *d)?;
        Some(
            self.bufs[at..]
                .iter()
                .flat_map(|(b, _)| b.iter().copied())
                .collect(),
        )
    }
}

struct RecSink {
    rec: Arc<Mutex<Recorded>>,
}

impl Element for RecSink {
    fn desc(&self) -> &'static ElementDesc {
        &REC_DESC
    }
    fn start(&mut self, _ctx: &mut Ctx) -> Result<(), Error> {
        Ok(())
    }
    fn process(&mut self, _ctx: &mut Ctx, mut inputs: Inputs<'_>) -> Result<Flow, Error> {
        let mut rec = self.rec.lock().unwrap();
        while let Some(buf) = inputs.pop() {
            rec.bufs.push((
                buf.memory.data().to_vec(),
                buf.flags.contains(BufferFlags::DISCONT),
            ));
        }
        Ok(Flow::Ok)
    }
    fn event(&mut self, _ctx: &mut Ctx, _event: &Event) -> Result<(), Error> {
        Ok(())
    }
    fn stop(&mut self, _ctx: &mut Ctx) {}
}

/// Run `httpsrc(url) ! recsink` to completion, returning what the sink recorded.
fn run_recorded(src: HttpSrc) -> (Result<(), Error>, Recorded) {
    let rec = Arc::new(Mutex::new(Recorded::default()));
    let mut p = Pipeline::new();
    let s = p.add(src);
    let sink = p.add(RecSink { rec: Arc::clone(&rec) });
    p.link((s, "src"), (sink, "sink")).expect("link");
    let result = p.run();
    let rec = Arc::try_unwrap(rec).unwrap().into_inner().unwrap();
    (result, rec)
}

// --- redirects (RFC 9110 §15.4) -------------------------------------------------------

#[test]
fn follows_a_redirect_chain_absolute_then_relative() {
    // 302 to an absolute URI, then 307 to a bare relative path, then the body. The
    // relative hop is the one RFC 3986 §5 reference resolution has to get right:
    // `third` against `/b/second` is `/b/third`, not `/third`.
    let body = ramp(200 * 1024);
    let served = body.clone();
    let port = Arc::new(AtomicU16::new(0));
    let their_port = Arc::clone(&port);
    let server = serve_script(move |_, head, s| {
        let p = their_port.load(Ordering::Acquire);
        match request_target(head) {
            "/a/start" => write_all(
                s,
                &format!(
                    "HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{p}/b/second\r\n\
                     Content-Length: 0\r\nConnection: close\r\n\r\n"
                ),
                b"",
            ),
            "/b/second" => write_all(
                s,
                "HTTP/1.1 307 Temporary Redirect\r\nLocation: third\r\n\
                 Content-Length: 0\r\nConnection: close\r\n\r\n",
                b"",
            ),
            "/b/third" => write_all(
                s,
                &format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    served.len()
                ),
                &served,
            ),
            other => write_all(
                s,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                other.as_bytes(),
            ),
        }
    });
    port.store(server.port, Ordering::Release);

    let (result, rec) = run_recorded(HttpSrc::new(server.url("/a/start")));
    result.expect("run");
    assert_eq!(rec.all(), body, "the body from the end of the chain arrives");

    let reqs = server.requests();
    assert_eq!(reqs.len(), 3, "three hops: {reqs:?}");
    assert_eq!(request_target(&reqs[0]), "/a/start");
    assert_eq!(request_target(&reqs[1]), "/b/second");
    assert_eq!(
        request_target(&reqs[2]),
        "/b/third",
        "a relative Location resolves against the request URI (RFC 3986 §5.2.3)"
    );
}

#[test]
fn redirect_hop_cap_errors_cleanly() {
    // A server that redirects forever. RFC 9110 §15.4: a client should detect and
    // intervene in cyclical redirections — here after ten hops, with an error rather
    // than a hang or a stack of connections.
    let server = serve_script(|_, _, s| {
        write_all(
            s,
            "HTTP/1.1 302 Found\r\nLocation: /next\r\nContent-Length: 0\r\n\
             Connection: close\r\n\r\n",
            b"",
        )
    });

    let (result, rec) = run_recorded(HttpSrc::new(server.url("/start")));
    let err = format!("{:?}", result.expect_err("a redirect loop must error"));
    assert!(err.contains("redirect"), "error names the cause: {err}");
    assert!(rec.all().is_empty(), "no body was emitted");
    assert!(
        settle(|| server.requests().len() >= 11),
        "the cap is hit, not exceeded silently"
    );
    assert_eq!(
        server.requests().len(),
        11,
        "one original request plus ten followed hops"
    );
}

#[test]
fn redirect_may_change_host_and_port() {
    // The `Location` moves the download to a different origin — the ordinary shape of
    // a podcast feed's enclosure URL handing off to a CDN. The second origin must see
    // its own `Host` (RFC 9112 §3.2), port included since it is not the default.
    let body = ramp(64 * 1024);
    let served = body.clone();
    let origin = serve_script(move |_, _, s| {
        write_all(
            s,
            &format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                served.len()
            ),
            &served,
        )
    });
    let origin_port = origin.port;
    let redirector = serve_script(move |_, _, s| {
        write_all(
            s,
            &format!(
                "HTTP/1.1 301 Moved Permanently\r\n\
                 Location: http://127.0.0.1:{origin_port}/media/ep1.mp3\r\n\
                 Content-Length: 0\r\nConnection: close\r\n\r\n"
            ),
            b"",
        )
    });

    let (result, rec) = run_recorded(HttpSrc::new(redirector.url("/feed/ep1")));
    result.expect("run");
    assert_eq!(rec.all(), body, "the body comes from the new origin");

    let at_origin = origin.requests();
    assert_eq!(at_origin.len(), 1, "the origin was asked exactly once");
    assert_eq!(request_target(&at_origin[0]), "/media/ep1.mp3");
    assert!(
        at_origin[0].contains(&format!("Host: 127.0.0.1:{origin_port}")),
        "Host follows the redirect, port included: {:?}",
        at_origin[0].lines().next()
    );
}

// --- resume after a mid-body disconnect (RFC 9110 §14.2) ------------------------------

#[test]
fn resumes_with_a_range_after_a_mid_body_disconnect() {
    // The server promises `Content-Length`, sends half, and hangs up. The client must
    // come back asking for exactly the bytes it did not get — and, because the first
    // response carried a strong `ETag`, must make that request conditional (§14.5).
    let body = ramp(300 * 1024);
    let half = body.len() / 2;
    let served = body.clone();
    let server = serve_script(move |n, head, s| {
        let total = served.len();
        if n == 0 {
            write_all(
                s,
                &format!(
                    "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nETag: \"v1\"\r\n\
                     Content-Length: {total}\r\nConnection: close\r\n\r\n"
                ),
                &served[..half],
            );
            return; // drop the connection with half the body still owed
        }
        assert!(
            head.contains(&format!("Range: bytes={half}-")),
            "the resume asks for the remainder: {head:?}"
        );
        write_all(
            s,
            &format!(
                "HTTP/1.1 206 Partial Content\r\n\
                 Content-Range: bytes {half}-{}/{total}\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                total - 1,
                total - half
            ),
            &served[half..],
        );
    });

    let (result, rec) = run_recorded(HttpSrc::new(server.url("/ep1.mp3")));
    result.expect("run");
    let got = rec.all();
    assert_eq!(got.len(), body.len(), "the whole body arrives exactly once");
    assert_eq!(got, body, "the two halves splice byte-exactly");
    assert!(
        !rec.bufs.iter().any(|(_, d)| *d),
        "a resume is contiguous — no DISCONT on a stream that never jumped"
    );

    let reqs = server.requests();
    assert_eq!(reqs.len(), 2, "one original request, one resume");
    assert!(
        reqs[1].contains("If-Range: \"v1\""),
        "the resume is conditional on the first response's strong validator: {:?}",
        reqs[1]
    );
}

#[test]
fn a_ranged_request_answered_200_fails_instead_of_splicing() {
    // The origin ignores `Range` and starts the representation over. Byte `half` of the
    // download is not byte `half` of that response, so continuing would duplicate the
    // first half into the middle of the file. RFC 9110 §15.3.7: 206 is the answer to a
    // satisfiable range request; a 200 is the server saying it will not do ranges.
    let body = ramp(200 * 1024);
    let half = body.len() / 2;
    let served = body.clone();
    let server = serve_script(move |n, _, s| {
        let total = served.len();
        let head = format!(
            "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\n\
             Connection: close\r\n\r\n"
        );
        if n == 0 {
            write_all(s, &head, &served[..half]);
        } else {
            write_all(s, &head, &served); // the whole thing again, from byte 0
        }
    });

    let (result, rec) = run_recorded(HttpSrc::new(server.url("/ep1.mp3")));
    let err = format!("{:?}", result.expect_err("a 200 to a Range must not splice"));
    assert!(
        err.contains("byte ranges") && err.contains("Range: bytes="),
        "the error says what the server did: {err}"
    );
    assert_eq!(
        rec.all(),
        body[..half],
        "only the bytes that were genuinely delivered were emitted"
    );
}

#[test]
fn a_stalled_socket_reconnects_within_budget() {
    // The server accepts, sends half the body, and then says nothing at all — the case
    // that used to hang the group thread forever, because the reactor has no timeout
    // op. `SO_RCVTIMEO` turns the dead air into a failed read, which lands in the
    // resume path.
    let body = ramp(120 * 1024);
    let half = body.len() / 2;
    let served = body.clone();
    let server = serve_script(move |n, _, s| {
        let total = served.len();
        if n == 0 {
            write_all(
                s,
                &format!(
                    "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\n\
                     Connection: close\r\n\r\n"
                ),
                &served[..half],
            );
            // Hold the connection open and silent. Long enough that the test can only
            // pass by timing out and reconnecting.
            std::thread::sleep(Duration::from_secs(60));
            return;
        }
        write_all(
            s,
            &format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {half}-{}/{total}\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                total - 1,
                total - half
            ),
            &served[half..],
        );
    });

    let began = Instant::now();
    let (result, rec) = run_recorded(
        HttpSrc::new(server.url("/ep1.mp3")).stall_timeout(Duration::from_millis(400)),
    );
    result.expect("run");
    assert_eq!(rec.all(), body, "the download completed across the stall");
    // 400 ms of dead air + a 500 ms backoff + the transfer. A generous ceiling: what
    // this asserts is that it finished at all, and nowhere near the 60 s the server
    // sat silent for.
    assert!(
        began.elapsed() < Duration::from_secs(10),
        "recovered promptly (took {:?})",
        began.elapsed()
    );
    assert_eq!(server.requests().len(), 2, "one stalled attempt, one resume");
}

// --- seeking (RFC 9110 §14) -----------------------------------------------------------

#[test]
fn seek_reopens_the_stream_with_a_range_request() {
    // A seek mid-download: the connection is torn down and reopened with
    // `Range: bytes=<target>-`, and the bytes that come out downstream from the DISCONT
    // marker on are exactly the tail of the resource from that offset.
    let body = ramp(400 * 1024);
    let target = 300 * 1024usize;
    let served = body.clone();
    let server = serve_script(move |n, head, s| {
        let total = served.len();
        if n == 0 {
            // Dribble, so the test thread has time to seek into the middle of it.
            write_slow(
                s,
                &format!(
                    "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {total}\r\n\
                     Connection: close\r\n\r\n"
                ),
                &served,
                8 * 1024,
                Duration::from_millis(4),
            );
            return;
        }
        assert!(
            head.contains(&format!("Range: bytes={target}-")),
            "the seek asks for the tail from the target: {head:?}"
        );
        write_all(
            s,
            &format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {target}-{}/{total}\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n",
                total - 1,
                total - target
            ),
            &served[target..],
        );
    });

    let rec = Arc::new(Mutex::new(Recorded::default()));
    let mut p = Pipeline::new();
    let src = HttpSrc::new(server.url("/ep1.mp3"));
    let info = src.info();
    let s = p.add(src);
    let sink = p.add(RecSink { rec: Arc::clone(&rec) });
    p.link((s, "src"), (sink, "sink")).expect("link");
    let seek = p.seek_handle();

    let seen = Arc::clone(&rec);
    let run = std::thread::spawn(move || p.run());
    // Wait until the download is genuinely under way before seeking, so the seek lands
    // mid-stream rather than racing `start()`.
    assert!(
        settle(|| seen.lock().unwrap().all().len() >= 32 * 1024),
        "the download started"
    );
    assert!(
        info.is_seekable(),
        "Accept-Ranges + Content-Length means seekable"
    );
    assert_eq!(info.total_bytes(), Some(body.len() as u64));
    seek.seek(target as u64, Timestamp::ZERO);
    run.join().expect("joined").expect("run");

    drop(seen); // the polling handle, so the recording can be taken back out
    let rec = Arc::try_unwrap(rec).unwrap().into_inner().unwrap();
    let tail = rec
        .after_discont()
        .expect("the first post-seek buffer is flagged DISCONT");
    assert_eq!(
        tail.len(),
        body.len() - target,
        "the post-seek stream is the whole tail"
    );
    assert_eq!(tail, body[target..], "post-seek bytes are at the right offset");
    assert_eq!(server.requests().len(), 2, "one original request, one seek");
}

#[test]
fn a_seek_on_an_unseekable_stream_warns_and_keeps_streaming() {
    // No `Accept-Ranges`: the seek cannot be honoured. Refusing it must not kill the
    // download — the app was told (via `HttpInfo::is_seekable`) that this stream does
    // not seek, and a chunked live feed is still perfectly playable straight through.
    let body = ramp(400 * 1024);
    let served = body.clone();
    let server = serve_script(move |_, _, s| {
        write_slow(
            s,
            &format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                served.len()
            ),
            &served,
            8 * 1024,
            Duration::from_millis(4),
        )
    });

    let rec = Arc::new(Mutex::new(Recorded::default()));
    let mut p = Pipeline::new();
    let src = HttpSrc::new(server.url("/live.mp3"));
    let info = src.info();
    let s = p.add(src);
    let sink = p.add(RecSink { rec: Arc::clone(&rec) });
    p.link((s, "src"), (sink, "sink")).expect("link");
    let seek = p.seek_handle();

    let seen = Arc::clone(&rec);
    // The pipeline rides back out of the thread so its bus can be drained afterwards.
    let run = std::thread::spawn(move || {
        let result = p.run();
        (result, p)
    });
    assert!(
        settle(|| seen.lock().unwrap().all().len() >= 32 * 1024),
        "the download started"
    );
    assert!(!info.is_seekable(), "no Accept-Ranges means not seekable");
    seek.seek(200 * 1024, Timestamp::ZERO);
    let (result, p) = run.join().expect("joined");
    result.expect("a refused seek must not fail the run");

    drop(seen);
    let rec = Arc::try_unwrap(rec).unwrap().into_inner().unwrap();
    let got = rec.all();
    // The flush itself still drops whatever was in flight when it landed (that is what
    // a flush *is*, and the scheduler's generation stamp discards it downstream), so
    // the recording may be a few buffers short. What must hold is that the source read
    // on to the true end of the resource, at the right offsets, and never jumped.
    assert!(
        got.len() <= body.len() && got.len() + 256 * 1024 >= body.len(),
        "streaming continued past the refused seek: {} of {} bytes",
        got.len(),
        body.len()
    );
    assert_eq!(
        got[got.len() - 32 * 1024..],
        body[body.len() - 32 * 1024..],
        "the stream ran to the real end of the resource"
    );
    assert!(!rec.bufs.iter().any(|(_, d)| *d), "nothing was repositioned");
    assert_eq!(
        server.requests().len(),
        1,
        "no pointless reconnect was attempted"
    );
    let mut warned = false;
    while let Some(msg) = p.bus().try_recv() {
        if let BusMessage::Warning { error, .. } = msg {
            warned |= format!("{error:?}").contains("ignoring seek");
        }
    }
    assert!(warned, "the refusal is reported on the bus");
}

#[test]
fn seek_to_zero_restarts_even_without_range_support() {
    // Byte zero is the one offset every server can serve without ranges: it is just the
    // request again. So a seek back to the start is honoured on a stream that cannot
    // seek anywhere else — and it goes out as a plain GET, not as `Range: bytes=0-`,
    // which an origin that ignores ranges would answer identically but which would
    // leave us unable to tell the two cases apart.
    let body = ramp(400 * 1024);
    let served = body.clone();
    let server = serve_script(move |_, _, s| {
        write_slow(
            s,
            &format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                served.len()
            ),
            &served,
            8 * 1024,
            Duration::from_millis(4),
        )
    });

    let rec = Arc::new(Mutex::new(Recorded::default()));
    let mut p = Pipeline::new();
    let src = HttpSrc::new(server.url("/ep1.mp3"));
    let info = src.info();
    let s = p.add(src);
    let sink = p.add(RecSink { rec: Arc::clone(&rec) });
    p.link((s, "src"), (sink, "sink")).expect("link");
    let seek = p.seek_handle();

    let seen = Arc::clone(&rec);
    let run = std::thread::spawn(move || p.run());
    assert!(
        settle(|| seen.lock().unwrap().all().len() >= 32 * 1024),
        "the download started"
    );
    assert!(!info.is_seekable(), "no Accept-Ranges means no seeking");
    seek.seek(0, Timestamp::ZERO);
    run.join().expect("joined").expect("run");

    drop(seen);
    let rec = Arc::try_unwrap(rec).unwrap().into_inner().unwrap();
    let restarted = rec
        .after_discont()
        .expect("the restart is marked DISCONT — the stream jumped backwards");
    assert_eq!(restarted, body, "the whole resource arrives again from byte 0");

    let reqs = server.requests();
    assert_eq!(reqs.len(), 2, "one original request, one restart");
    assert!(
        !reqs[1].contains("Range:"),
        "a restart at zero needs no Range header: {:?}",
        reqs[1]
    );
}

#[test]
fn content_length_and_range_support_reach_the_app() {
    // The byte "duration" a controller needs for a proportional seek. `filesrc`'s
    // equivalent is a `stat` the app does itself; over HTTP only the element ever sees
    // the response head, so it hands the facts back through `HttpInfo`.
    let body = ramp(50 * 1024);
    let served = body.clone();
    let server = serve_script(move |_, _, s| {
        write_all(
            s,
            &format!(
                "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n",
                served.len()
            ),
            &served,
        )
    });

    let src = HttpSrc::new(server.url("/ep1.mp3"));
    let info = src.info();
    assert_eq!(info.total_bytes(), None, "nothing is known before start()");
    assert!(!info.is_seekable());

    let (result, rec) = run_recorded(src);
    result.expect("run");
    assert_eq!(rec.all().len(), body.len());
    assert_eq!(info.total_bytes(), Some(body.len() as u64));
    assert!(info.is_seekable());

    // A chunked response states no length, so it must not claim to be seekable.
    let encoded = chunked_encode(&ramp(4096), &[1000, 2000]);
    let chunked = serve_script(move |_, _, s| {
        write_all(
            s,
            "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nTransfer-Encoding: chunked\r\n\
             Connection: close\r\n\r\n",
            &encoded,
        )
    });
    let src = HttpSrc::new(chunked.url("/live.mp3"));
    let info = src.info();
    let (result, _) = run_recorded(src);
    result.expect("run");
    assert_eq!(info.total_bytes(), None, "chunked declares no length");
    assert!(!info.is_seekable(), "no length means no seeking");
}

/// A real fetch over the public internet: one GET of the first megabyte of a podcast
/// enclosure, through the tracking-redirect chain such a URL always has. Ignored by
/// default — it needs the network, and the URL is somebody else's episode, which will
/// eventually rotate out of the feed — but it is the check that redirects, TLS and the
/// length plumbing work against real CDNs rather than against this file's fixtures.
///
/// Run with: `cargo test -p pf-http --test download -- --ignored real_world`
#[test]
#[ignore = "needs the public internet"]
fn real_world_podcast_enclosure_first_megabyte() {
    // From NPR's Planet Money feed (`https://feeds.npr.org/510289/podcast.xml`): the
    // enclosure is a Swap.fm tracking URL that hands off to Spotify's prefix service,
    // then Podtrac, then the Simplecast origin — four hops, https throughout.
    const URL: &str = "https://tracking.swap.fm/track/XvDEoI11TR00olTUO8US/prfx.byspotify.com/e/play.podtrac.com/npr-510289/npr.simplecastaudio.com/7dda0db1-b7e8-490c-b09a-f22fdeb30a87/episodes/fda39dfe-a1eb-40f3-ab73-8b6a189001f1/audio/128/default.mp3?awCollectionId=7dda0db1-b7e8-490c-b09a-f22fdeb30a87&awEpisodeId=fda39dfe-a1eb-40f3-ab73-8b6a189001f1&feed=1e5vv9pg&t=podcast";

    let rec = Arc::new(Mutex::new(Recorded::default()));
    let mut p = Pipeline::new();
    let src = HttpSrc::new(URL).stall_timeout(Duration::from_secs(20));
    let info = src.info();
    let s = p.add(src);
    let sink = p.add(RecSink { rec: Arc::clone(&rec) });
    p.link((s, "src"), (sink, "sink")).expect("link");
    let stop = p.stop_handle();

    let seen = Arc::clone(&rec);
    let run = std::thread::spawn(move || p.run());
    // A megabyte is the spot check; hang up rather than pull down a whole episode.
    let enough = settle_within(Duration::from_secs(60), || {
        seen.lock().unwrap().all().len() >= 1024 * 1024
    });
    stop.stop();
    let result = run.join().expect("joined");

    drop(seen);
    let rec = Arc::try_unwrap(rec).unwrap().into_inner().unwrap();
    let got = rec.all();
    println!(
        "real-world: {} bytes taken, Content-Length {:?}, seekable {}, run {result:?}",
        got.len(),
        info.total_bytes(),
        info.is_seekable()
    );
    result.expect("run");
    assert!(enough, "a megabyte arrived (got {} bytes)", got.len());
    assert!(
        info.total_bytes().is_some_and(|n| n > 1024 * 1024),
        "the episode's length came back from the origin: {:?}",
        info.total_bytes()
    );
    // An MP3 enclosure starts with an ID3 tag or an MPEG frame sync.
    assert!(
        got.starts_with(b"ID3") || (got[0] == 0xFF && got[1] & 0xE0 == 0xE0),
        "looks like an MP3: {:02x?}",
        &got[..4]
    );
}
