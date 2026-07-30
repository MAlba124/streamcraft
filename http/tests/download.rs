//! Milestone 2: `httpsrc ! filesink` downloads a file over HTTP and writes it to
//! disk byte-identically (spec: Milestone applications §2).
//!
//! Hermetic — no real internet. Each test stands up a `std::net::TcpListener` on
//! `127.0.0.1:0`, serves one canned response from a background thread, and points the
//! pipeline at it.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

use pf_http::HttpSrc;
use profluens_core::pipeline::Pipeline;
use profluens_elements::io::FileSink;

fn temp_path(tag: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("pf_m2_{}_{}.bin", tag, std::process::id()));
    p
}

/// Read a full HTTP request head (up to and including the blank line) off a stream.
fn read_request(stream: &mut std::net::TcpStream) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 512];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(_) => break,
        }
    }
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
